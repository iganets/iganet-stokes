/**
   @file examples/iganet_fitting_simple.cxx

   @brief Demonstration of IgANet function fitting

   This example demonstrates how to implement a simple IgANet to fit a
   given function on a square geometry. In contrast to the example
   iganet_fitting.cxx this examples does not make use of pre-computed
   indices and coefficients and might therefore be slower.

   This example can be configured with the following environment variables

   IGANET_NCOEFFS   - number of B-spline coefficients
   IGANET_NLAYERS   - number of network layers
   IGANET_NNEURONS  - number of neurons per layer
   IGANET_MAX_EPOCH - maximum number of epochs during training
   IGANET_MIN_LOSS  - tolerance for loss function

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <chrono>
#include <iganet.h>
#include <iostream>

/// @brief Specialization of the abstract IgANet class for function fitting
template <typename Optimizer, typename GeometryMap, typename Variable>
class fitting : public iganet::IgANet<Optimizer, GeometryMap, Variable,
                                      iganet::IgABaseNoRefData> {

private:
  /// @brief Type of the base class
  using Base = iganet::IgANet<Optimizer, GeometryMap, Variable,
                              iganet::IgABaseNoRefData>;

  /// @brief Collocation points
  typename Base::variable_collPts_type collPts_;

public:
  /// @brief Constructors from the base class
  using iganet::IgANet<Optimizer, GeometryMap, Variable,
                       iganet::IgABaseNoRefData>::IgANet;

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Initializes the epoch
  ///
  /// @param[in] epoch Epoch number
  bool epoch(int64_t epoch) override {
    // In the very first epoch we need to generate the sampling points
    // for the inputs and the sampling points in the function space of
    // the variables since otherwise the respective tensors would be
    // empty. In all further epochs no updates are needed since we do
    // not change the inputs nor the variable function space.
    if (epoch == 0) {
      Base::inputs(epoch);
      collPts_ = Base::variable_collPts(iganet::collPts::greville);

      return true;
    } else
      return false;
  }

  /// @brief Computes the loss function
  ///
  /// @param[in] outputs Output of the network
  ///
  /// @param[in] epoch Epoch number
  torch::Tensor loss(const torch::Tensor &outputs, int64_t epoch) override {

    // Cast the network output (a raw tensor) into the proper
    // function-space format, i.e. B-spline objects for the interior
    // and boundary parts that can be evaluated.
    Base::u_.from_tensor(outputs);

    // Evaluate the loss function
    return torch::mse_loss(*Base::u_.eval(collPts_.first)[0],
                           sin(M_PI * collPts_.first[0]) *
                               sin(M_PI * collPts_.first[1]));
  }
};

int main() {
  iganet::init();
  iganet::verbose(std::cout);

  nlohmann::json json;
  json["res0"] = 50;
  json["res1"] = 50;

  using namespace iganet::literals;
  using optimizer_t = torch::optim::Adam;
  using real_t = double;

  // Geometry: Bi-linear B-spline function space S (geoDim = 2, p = q = 1)
  using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 2, 1, 1>>;

  // Variable: Bi-quadratic B-spline function space S (geoDim = 1, p = q = 2)
  using variable_t = iganet::S<iganet::UniformBSpline<real_t, 1, 2, 2>>;

  // Loop over user-definded number of coefficients (default 32)
  for (int64_t ncoeffs : iganet::utils::getenv("IGANET_NCOEFFS", {32})) {
    // Loop over different activation functions (default ReLU)
    for (std::vector<std::any> activation :
         {std::vector<std::any>{iganet::activation::relu}}) {
      // Loop over user-defined numbers of layers (default 1)
      for (int64_t nlayers : iganet::utils::getenv("IGANET_NLAYERS", {1})) {
        // Loop over user-defined number of neurons per layer (default 10)
        for (int64_t nneurons :
             iganet::utils::getenv("IGANET_NNEURONS", {10})) {

          std::vector<int64_t> layers(nlayers, nneurons);
          std::vector<std::vector<std::any>> activations(nlayers, activation);
          activations.emplace_back(
              std::vector<std::any>{iganet::activation::none});

          fitting<optimizer_t, geometry_t, variable_t>
              net( // Number of neurons per layers
                  layers,
                  // Activation functions
                  activations,
                  // Number of B-spline coefficients of the geometry, just [0,1]
                  // x [0,1]
                  iganet::utils::to_array(2_i64, 2_i64),
                  // Number of B-spline coefficients of the variable
                  iganet::utils::to_array(ncoeffs, ncoeffs));

          iganet::Log(iganet::log::info)
              << "#coeff: " << ncoeffs << ", #layers: " << nlayers
              << ", #neurons: " << nneurons
              << ", #parameters: " << net.nparameters() << std::endl;

          // Set maximum number of epochs
          net.options().max_epoch(
              iganet::utils::getenv("IGANET_MAX_EPOCH", 1000_i64));

          // Set tolerance for the loss functions
          net.options().min_loss(
              iganet::utils::getenv("IGANET_MIN_LOSS", 1e-12));

          // Start time measurement
          auto t1 = std::chrono::high_resolution_clock::now();

          // Train network
          net.train();

          // Stop time measurement
          auto t2 = std::chrono::high_resolution_clock::now();
          iganet::Log(iganet::log::info)
              << "Training took "
              << std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
                                                                           t1)
                     .count()
              << " seconds\n";

#ifdef IGANET_WITH_MATPLOT
          // Plot the solution
          net.G()
              .space()
              .plot(net.u().space(), net.collPts().first, json)
              ->show();
#endif

#ifdef IGANET_WITH_GISMO
          // Convert B-spline objects to G+Smo
          auto G_gismo = net.G().space().to_gismo();
          auto u_gismo = net.u().space().to_gismo();
          gsFunctionExpr<real_t> f_gismo("sin(pi*x)*sin(pi*y)", 2);

          // Set up expression assembler
          gsExprAssembler<real_t> A(1, 1);
          gsMultiBasis<real_t> basis(u_gismo, true);

          A.setIntegrationElements(basis);

          auto G = A.getMap(G_gismo);
          auto u = A.getCoeff(u_gismo, G);
          auto f = A.getCoeff(f_gismo, G);

          // Compute L2- and H2-error
          gsExprEvaluator<real_t> ev(A);

          iganet::Log(iganet::log::info)
              << "L2-error : "
              << gismo::math::sqrt(ev.integral((u - f).sqNorm() * meas(G)))
              << std::endl;

          iganet::Log(iganet::log::info)
              << "H1-error : "
              << gismo::math::sqrt(ev.integral(
                     (gismo::expr::igrad(u, G) - gismo::expr::igrad(f, G))
                         .sqNorm() *
                     meas(G)))
              << std::endl;
#endif
        }
      }
    }
  }

  iganet::finalize();
  return 0;
}
