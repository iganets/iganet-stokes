/**
   @file examples/iganet_fitting_geometry.cxx

   @brief Demonstration of IgANet function fitting on a geometry loaded from a
   file

   @author Veronika Travnikova

   This example demonstrates how to implement an IgANet to fit a given
   function on a geometry loaded from a file. In contrast to the
   example iganet_fitting_geometry_simple.cxx this examples makes use
   of pre-computed indices and coefficients and should therefore be
   faster.

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
    : public iganet::IgANet<Optimizer, GeometryMap, Variable,
                            iganet::IgABaseNoRefData>,
      public iganet::IgANetCustomizable<GeometryMap, Variable> {

private:
  /// @brief Type of the base class
  using Base = iganet::IgANet<Optimizer, GeometryMap, Variable,
                              iganet::IgABaseNoRefData>;

  /// @brief Collocation points
  typename Base::variable_collPts_type collPts_;

  /// @brief Type of the customizable class
   /// @brief Type of the customizable class
  using Customizable = iganet::IgANetCustomizable<GeometryMap, Variable>;

  /// @brief Knot indices
  typename Customizable::variable_interior_knot_indices_type knot_indices_;

  /// @broef Coefficient indices
  typename Customizable::variable_interior_coeff_indices_type coeff_indices_;

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
          Base::u_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      coeff_indices_ =
          Base::u_.template find_coeff_indices<iganet::functionspace::interior>(
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
        sin(M_PI * collPts_.first[0]) * sin(M_PI * collPts_.first[1]));
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

  pugi::xml_document xml;
  xml.load_file(IGANET_DATA_DIR "surfaces/2d/quadCircleNUB_ref.xml");


  // Bivariate uniform B-spline of degree 2 in both directions
  // the type has to correspond to the respective geometry parameterization in
  // the input file
  using geometry_t = iganet::S<iganet::NonUniformBSpline<real_t, 2, 2, 3>>;

  // Variable: Bi-quadratic B-spline function space S2 (geoDim = 1, p = q = 2)
  using variable_t = iganet::S<iganet::UniformBSpline<real_t, 1, 2, 2>>;

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
          // Number of B-spline coefficients of the geometry, has to correspond
          // to number of coefficients in input file
          std::tuple(iganet::utils::to_array(4_i64, 25_i64)));

  // Load geometry parameterization from XML
  net.G().from_xml(xml);

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
  // Evaluate position of collocation points in physical domain
  //auto f_plot = sin(M_PI * net.collPts().first[0]) * sin(M_PI * net.collPts().first[1]);

  // Plot the solution
  net.G().space().plot(net.u().space(), net.collPts().first, json)->show();


  // Plot the difference between the solution and the reference data
  //net.G().space()
  //    .plot(net.u().space().abs_diff(f_plot), json)
  //    ->show();
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
             (gismo::expr::igrad(u, G) - gismo::expr::igrad(f, G)).sqNorm() *
             meas(G)))
      << std::endl;
#endif

  return 0;
}
