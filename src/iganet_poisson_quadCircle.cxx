/**
   @file examples/iganet_poisson.cxx

   @brief Demonstration of IgANet Poisson solver

   This example demonstrates how to implement a simple IgANet for
   learning the Poisson equation with (non-)homogeneous Dirichlet
   boundary conditions on a square geometry.

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <iostream>

using namespace iganet::literals;

/// @brief Specialization of the abstract IgANet class for Poisson's equation
template <typename Optimizer, typename GeometryMap, typename Variable>
class poisson
    : public iganet::IgANet<Optimizer, GeometryMap, Variable>,
      public iganet::IgANetCustomizable<GeometryMap, Variable> {

private:
  /// @brief Type of the base class
  using Base = iganet::IgANet<Optimizer, GeometryMap, Variable>;

  /// @brief Collocation points
  typename Base::variable_collPts_type collPts_;

  /// @brief Reference solution
  Variable ref_;

  /// @brief Type of the customizable class
  using Customizable =
      iganet::IgANetCustomizable<GeometryMap, Variable>;

  /// @brief Knot indices of variables
  typename Customizable::variable_interior_knot_indices_type var_knot_indices_;

  /// @broef Coefficient indices of variables
  typename Customizable::variable_interior_coeff_indices_type
      var_coeff_indices_;

  /// @brief Knot indices of the geometry map
  typename Customizable::geometryMap_interior_knot_indices_type G_knot_indices_;

  /// @broef Coefficient indices of the geometry map
  typename Customizable::geometryMap_interior_coeff_indices_type
      G_coeff_indices_;

public:
  /// @brief Constructor
  template <typename... Args>
  poisson(std::vector<int64_t> &&layers,
          std::vector<std::vector<std::any>> &&activations, Args &&...args)
      : Base(std::forward<std::vector<int64_t>>(layers),
             std::forward<std::vector<std::vector<std::any>>>(activations),
             std::forward<Args>(args)...),
        ref_(iganet::utils::to_array(4_i64, 33_i64)) {}

  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }

  /// @brief Returns a constant reference to the reference solution
  auto const &ref() const { return ref_; }

  /// @brief Returns a non-constant reference to the reference solution
  auto &ref() { return ref_; }

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

      var_knot_indices_ =
          Base::f_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      var_coeff_indices_ =
          Base::f_.template find_coeff_indices<iganet::functionspace::interior>(
              var_knot_indices_);

      G_knot_indices_ =
          Base::G_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      G_coeff_indices_ =
          Base::G_.template find_coeff_indices<iganet::functionspace::interior>(
              G_knot_indices_);

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

    // Evaluate the Laplacian operator
    //auto u_ilapl =
    //    Base::u_.ilapl(Base::G_, collPts_.first, var_knot_indices_,
    //                   var_coeff_indices_, G_knot_indices_, G_coeff_indices_);

    auto u_ilapl = Base::u_.ilapl(Base::G_, collPts_.first, var_knot_indices_,
                       var_coeff_indices_, G_knot_indices_, G_coeff_indices_);
    auto f =
        Base::f_.eval(collPts_.first, var_knot_indices_, var_coeff_indices_);

    auto u_bdr = Base::u_.template eval<iganet::functionspace::boundary>(
        collPts_.second);

    auto bdr =
        ref_.template eval<iganet::functionspace::boundary>(collPts_.second);

    // Evaluate the loss function
    auto loss = torch::mse_loss(*u_ilapl[0], *f[0]) +
           1e1 * torch::mse_loss(*std::get<0>(u_bdr)[0], *std::get<0>(bdr)[0]) +
           1e1 * torch::mse_loss(*std::get<1>(u_bdr)[0], *std::get<1>(bdr)[0]) +
           1e1 * torch::mse_loss(*std::get<2>(u_bdr)[0], *std::get<2>(bdr)[0]) +
           1e1 * torch::mse_loss(*std::get<3>(u_bdr)[0], *std::get<3>(bdr)[0]);
    return loss;
  }
};

int main() {
  iganet::init();
  iganet::verbose(iganet::Log(iganet::log::info));

  nlohmann::json json;
  json["res0"] = 50;
  json["res1"] = 50;

  using namespace iganet::literals;
  using optimizer_t = torch::optim::LBFGS;
  using real_t = double;

  
   // Load XML file
  pugi::xml_document xml;
  xml.load_file(IGANET_DATA_DIR "surfaces/2d_quadCircle_degree34/quadCircleImp_R1I04_resultR1E2Fixed.xml");

  using geometry_t = iganet::S<iganet::NonUniformBSpline<real_t, 2, 3, 4>>;
  using variable_t = iganet::S<iganet::UniformBSpline<real_t, 1, 3, 3>>;

  poisson<optimizer_t, geometry_t, variable_t>
      net( // Number of neurons per layers
          {700, 700, 700, 700},
          // Activation functions
          {{iganet::activation::sigmoid},
           {iganet::activation::sigmoid},
           {iganet::activation::sigmoid},
           {iganet::activation::sigmoid},
           {iganet::activation::none}},
          // Number of B-spline coefficients of the geometry, just [0,1] x [0,1]
          std::tuple(iganet::utils::to_array(5_i64, 41_i64)));

  // load geometry from file
  net.G().from_xml(xml);
  net.G().uniform_refine(1, 0);
  net.f().uniform_refine(1, 0);

  // Impose the negative of the second derivative of sin(M_PI*x) *
  // sin(M_PI*y) as right-hand side vector (manufactured solution)
  net.f().transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 1>{-2.0 * M_PI * M_PI * sin(M_PI * xi[0]) *
                                 sin(M_PI * xi[1])};
  });

  // Impose reference solution
  net.ref().transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 1>{sin(M_PI * xi[0]) * sin(M_PI * xi[1])};
  });

  // Impose boundary conditions
  net.ref().boundary().template side<1>().transform(
      [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0};
      });

  net.ref().boundary().template side<2>().transform(
      [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0};
      });

  net.ref().boundary().template side<3>().transform(
      [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0};
      });

  net.ref().boundary().template side<4>().transform(
      [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0};
      });

  // Set maximum number of epoches
  net.options().max_epoch(2000);

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
auto colPts = net.G().eval(net.collPts().first);
  // Plot the solution
    net.G()
      .plot(net.u(), std::array<torch::Tensor, 2>{*colPts[0], *colPts[1]}, json)
      ->show();

  // Plot the difference between the exact and predicted solutions
  net.G()
      .plot(net.u().abs_diff(net.ref()), std::array<torch::Tensor, 2>{*colPts[0], *colPts[1]},
           json)
      ->show();
#endif

  return 0;
}
