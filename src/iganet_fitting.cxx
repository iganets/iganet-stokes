/**
   @file examples/iganet_fitting.cxx

   @brief Demonstration of IgANet function fitting

   This example demonstrates how to implement an IgANet to fit a
   given function on a square geometry. In contrast to the example
   iganet_fitting_simple.cxx this examples makes use of pre-computed
   indices and coefficients and should therefore be faster.

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
class fitting
    : public iganet::IgANet<Optimizer, GeometryMap, Variable>,
      public iganet::IgANetCustomizable<Optimizer, GeometryMap, Variable> {

private:
  /// @brief Type of the base class
  using Base = iganet::IgANet<Optimizer, GeometryMap, Variable>;

  /// @brief Collocation points
  typename Base::variable_collPts_type collPts_;

  /// @brief Type of the customizable class
  using Customizable =
      iganet::IgANetCustomizable<Optimizer, GeometryMap, Variable>;

  /// @brief Knot indices
  typename Customizable::variable_interior_knot_indices_type knot_indices_;

  /// @broef Coefficient indices
  typename Customizable::variable_interior_coeff_indices_type coeff_indices_;

public:
  /// @brief Constructors from the base class
  using iganet::IgANet<Optimizer, GeometryMap, Variable>::IgANet;

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Initializes the epoch
  ///
  /// @param[in] epoch Epoch number
  bool epoch(int64_t epoch) override {
    std::clog << "Epoch " << std::to_string(epoch) << ": ";

    // In the very first epoch we need to generate the sampling points
    // for the inputs and the sampling points in the function space of
    // the variables since otherwise the respective tensors would be
    // empty. In all further epochs no updates are needed since we do
    // not change the inputs nor the variable function space.
    if (epoch == 0) {
      Base::inputs(epoch);
      collPts_ = Base::variable_collPts(iganet::collPts::greville);

      knot_indices_ =
          Base::f_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      coeff_indices_ =
          Base::f_.template find_coeff_indices<iganet::functionspace::interior>(
              knot_indices_);

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
    return torch::mse_loss(
        *Base::u_.eval(collPts_.first, knot_indices_, coeff_indices_)[0],
        *Base::f_.eval(collPts_.first, knot_indices_, coeff_indices_)[0]);
  }
};

int main() {
  iganet::init();
  iganet::verbose(iganet::Log(iganet::log::info));

  nlohmann::json json;
  json["res0"] = 50;
  json["res1"] = 50;

  using namespace iganet::literals;
  using optimizer_t = torch::optim::Adam;
  using real_t = double;

  // Geometry: Bi-linear B-spline function space S2 (geoDim = 2, p = q = 1)
  using geometry_t = iganet::S2<iganet::UniformBSpline<real_t, 2, 1, 1>>;

  // Variable: Bi-quadratic B-spline function space S2 (geoDim = 1, p = q = 2)
  using variable_t = iganet::S2<iganet::UniformBSpline<real_t, 1, 2, 2>>;

  fitting<optimizer_t, geometry_t, variable_t>
      net( // Number of neurons per layers
          {50, 50, 50, 50, 50},
          // Activation functions
          {{iganet::activation::sigmoid},
           {iganet::activation::sigmoid},
           {iganet::activation::sigmoid},
           {iganet::activation::sigmoid},
           {iganet::activation::sigmoid},
           {iganet::activation::none}},
          // Number of B-spline coefficients of the geometry, just [0,1] x [0,1]
          std::tuple(iganet::utils::to_array(2_i64, 2_i64)),
          // Number of B-spline coefficients of the variable
          std::tuple(iganet::utils::to_array(10_i64, 10_i64)));

  // Impose solution value for supervised training (not right-hand side)
  net.f().transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 1>{
        static_cast<real_t>(sin(M_PI * xi[0]) * sin(M_PI * xi[1]))};
  });

  // Set maximum number of epoches
  net.options().max_epoch(1000);

  // Set tolerance for the loss functions
  net.options().min_loss(1e-8);

  // Start time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  // Train network
  net.train();

  // Stop time measurement
  auto t2 = std::chrono::high_resolution_clock::now();
  iganet::Log(iganet::log::info)
      << "Training took "
      << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
             .count()
      << " seconds\n";

#ifdef IGANET_WITH_MATPLOT
  // Plot the solution
  net.G().plot(net.u(), net.collPts().first, json)->show();

  // Plot the difference between the solution and the reference data
  net.G().plot(net.u().abs_diff(net.f()), net.collPts().first, json)->show();
#endif

#ifdef IGANET_WITH_GISMO
  // Convert B-spline objects to G+Smo
  auto G_gismo = net.G().to_gismo();
  auto u_gismo = net.u().to_gismo();
  auto f_gismo = net.f().to_gismo();

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
             (gismo::expr::igrad(u, G) - gismo::expr::igrad(f, G)).sqNorm() *
             meas(G)))
      << std::endl;
#endif

  return 0;
}
