/**
   @file examples/iganet_fitting_geometry_autotune.cxx

   @brief Demonstration of IgANet function fitting on a geometry loaded from a
   file with automatic hyper-parameter tuning

   This example demonstrates how to auto-tune the hyper-parameters of
   an IgANet for fitting a given function on a geometry loaded from a file. In
   contrast to the example iganet_fitting_geometry_dataloader_autotone this
   example does not make use of the dataloader and fits only on a single
   geometry.

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
class fitting : public iganet::IgANet<Optimizer, GeometryMap, Variable>,
                public iganet::IgANetCustomizable<GeometryMap, Variable> {

private:
  /// @brief Type of the base class
  using Base = iganet::IgANet<Optimizer, GeometryMap, Variable>;

  /// @brief Collocation points
  typename Base::variable_collPts_type collPts_;

  /// @brief Type of the customizable class
  using Customizable = iganet::IgANetCustomizable<GeometryMap, Variable>;

  /// @brief Knot indices
  typename Customizable::variable_interior_knot_indices_type knot_indices_;

  /// @brief Coefficient indices
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
    // In the very first epoch we need to generate the sampling points
    // for the inputs and the sampling points in the function space of
    // the variables since otherwise the respective tensors would be
    // empty. In all further epochs no updates are needed since we do
    // not change the inputs nor the variable function space.
    if (epoch == 0) {
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
  iganet::verbose(std::cout);

  nlohmann::json json;
  json["res0"] = 50;
  json["res1"] = 50;

  using namespace iganet::literals;
  using optimizer_t = torch::optim::Adam;
  using real_t = double;

  // Load XML file
  pugi::xml_document xml;
  xml.load_file(IGANET_DATA_DIR "surfaces/2d/geo02.xml");

  // Bivariate uniform B-spline of degree 2 in both directions
  // the type has to correspond to the respective geometry parameterization in
  // the input file
  using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 2, 2, 2>>;

  // Variable: Bi-quadratic B-spline function space S (geoDim = 1, p = q = 2)
  using variable_t = iganet::S<iganet::UniformBSpline<real_t, 1, 2, 2>>;

  for (std::vector<std::any> activation :
       {std::vector<std::any>{iganet::activation::relu},
        std::vector<std::any>{iganet::activation::sigmoid},
        std::vector<std::any>{iganet::activation::tanh}}) {
    for (int64_t nlayers :
         iganet::utils::getenv("IGANET_NLAYERS", {5, 6, 7, 8, 9})) {
      for (int64_t nneurons : iganet::utils::getenv(
               "IGANET_NNEURONS", {60, 80, 100, 120, 140, 160, 180, 200})) {

        iganet::Log(iganet::log::info)
            << "#layers: " << nlayers << ", #neurons: " << nneurons
            << std::endl;

        std::vector<int64_t> layers(nlayers, nneurons);
        std::vector<std::vector<std::any>> activations(nlayers, activation);
        activations.emplace_back(
            std::vector<std::any>{iganet::activation::none});

        fitting<optimizer_t, geometry_t, variable_t>
            net( // Number of neurons per layers
                layers,
                // Activation functions
                activations,
                // Number of B-spline coefficients of the geometry
                std::tuple(iganet::utils::to_array(25_i64, 25_i64)),
                // Number of B-spline coefficients of the variable
                std::tuple(iganet::utils::to_array(30_i64, 30_i64)));

        // Load geometry parameterization from XML
        net.G().from_xml(xml);

        // Impose solution value for supervised training (not right-hand side)
        net.f().transform([](const std::array<real_t, 2> xi) {
          return std::array<real_t, 1>{
              static_cast<real_t>(sin(M_PI * xi[0]) * sin(M_PI * xi[1]))};
        });

        // Set maximum number of epochs
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
            << std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
                                                                         t1)
                   .count()
            << " seconds\n";
      }
    }
  }

  return 0;
}
