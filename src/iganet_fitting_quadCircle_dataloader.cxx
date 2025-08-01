// /**
//    @file examples/iganet_fitting_autotune.cxx

//    @brief Demonstration of IgANet function fitting with automatic
//    hyper-parameter tuning and use of the data loader for the geometry

//    This example demonstrates how to auto-tune the hyper-parameters of
//    an IgANet for fitting a given function on a set of geometries that
//    are loaded with the custom data loader

//    @author Veronika Travnikova

//    @copyright This file is part of the IgANet project

//    This Source Code Form is subject to the terms of the Mozilla Public
//    License, v. 2.0. If a copy of the MPL was not distributed with this
//    file, You can obtain one at http://mozilla.org/MPL/2.0/.
// */

// #include <chrono>
// #include <iganet.h>
// #include <iostream>

// /// @brief Specialization of the abstract IgANet class for function fitting
// template <typename Optimizer, typename GeometryMap, typename Variable>
// class fitting
//     : public iganet::IgANet<Optimizer, GeometryMap, Variable>,
//       public iganet::IgANetCustomizable<Optimizer, GeometryMap, Variable> {

// private:
//   /// @brief Type of the base class
//   using Base = iganet::IgANet<Optimizer, GeometryMap, Variable>;

//   /// @brief Collocation points
//   typename Base::variable_collPts_type collPts_;

//   /// @brief Type of the customizable class
//   using Customizable =
//       iganet::IgANetCustomizable<Optimizer, GeometryMap, Variable>;

//   /// @brief Knot indices
//   typename Customizable::variable_interior_knot_indices_type knot_indices_;

//   /// @broef Coefficient indices
//   typename Customizable::variable_interior_coeff_indices_type coeff_indices_;

// public:
//   /// @brief Constructors from the base class
//   using iganet::IgANet<Optimizer, GeometryMap, Variable>::IgANet;

//   /// @brief Returns a constant reference to the collocation points
//   auto const &collPts() const { return collPts_; }

//   /// @brief Initializes the epoch
//   ///
//   /// @param[in] epoch Epoch number
//   bool epoch(int64_t epoch) override {
//     // In the very first epoch we need to generate the sampling points
//     // for the inputs and the sampling points in the function space of
//     // the variables since otherwise the respective tensors would be
//     // empty. In all further epochs no updates are needed since we do
//     // not change the inputs nor the variable function space.
//     //    if (epoch == 0) {
//     collPts_ = Base::variable_collPts(iganet::collPts::greville_ref2);

//     knot_indices_ =
//         Base::f_.template find_knot_indices<iganet::functionspace::interior>(
//             collPts_.first);
//     coeff_indices_ =
//         Base::f_.template find_coeff_indices<iganet::functionspace::interior>(
//             knot_indices_);

//     return true;
//     //} else
//     // return false;
//   }

//   /// @brief Computes the loss function
//   ///
//   /// @param[in] outputs Output of the network
//   ///
//   /// @param[in] epoch Epoch number
//   torch::Tensor loss(const torch::Tensor &outputs, int64_t epoch) override {
//     // Cast the network output (a raw tensor) into the proper
//     // function-space format, i.e. B-spline objects for the interior
//     // and boundary parts that can be evaluated.
//     Base::u_.from_tensor(outputs.flatten());

//     // Evaluate the loss function
//     return torch::mse_loss(
//         *Base::u_.eval(collPts_.first, knot_indices_, coeff_indices_)[0],
//         *Base::f_.eval(collPts_.first, knot_indices_, coeff_indices_)[0]);
//   }
// };

 int main() {
//   iganet::init();
//   iganet::verbose(std::cout);

//   nlohmann::json json;
//   json["res0"] = 50;
//   json["res1"] = 50;

//   using namespace iganet::literals;
//   using optimizer_t = torch::optim::Adam;
//   using real_t = double;

//   // Geometry: Bi-linear B-spline function space S2 (geoDim = 2, p = q = 1)
//   using geometry_t = iganet::S2<iganet::NonUniformBSpline<real_t, 2, 2, 3>>;

//   // Variable: Bi-quadratic B-spline function space S2 (geoDim = 1, p = q = 2)
//   using variable_t = iganet::S2<iganet::UniformBSpline<real_t, 1, 2, 2>>;

//   // Data set for training
//   iganet::IgADataset<> data_set;
//   data_set.add_geometryMap(geometry_t{iganet::utils::to_array(4_i64, 25_i64)},
//                            IGANET_DATA_DIR "surfaces/2d_quadCircle_variableRadius");

//   // Impose solution value for supervised training (not right-hand side)
//   data_set.add_referenceData(
//       variable_t{iganet::utils::to_array(30_i64, 30_i64)},
//       [](const std::array<real_t, 2> xi) {
//         return std::array<real_t, 1>{
//             static_cast<real_t>(sin(M_PI * xi[0]) * sin(M_PI * xi[1]))};
//       });

//   auto train_set = data_set.map(
//       torch::data::transforms::Stack<iganet::IgADataset<>::example_type>());
//   auto train_size = train_set.size().value();
//   auto train_loader =
//       torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
//           std::move(train_set),
//           1); // at the moment the batch-size needs to be 1

//   fitting<optimizer_t, geometry_t, variable_t>
//       net( // Number of neurons per layers
//           {100, 100},
//           // Activation functions
//           {{iganet::activation::sigmoid},
//            {iganet::activation::sigmoid},
//            {iganet::activation::none}},
//           // Number of B-spline coefficients of the geometry
//           std::tuple(iganet::utils::to_array(4_i64, 25_i64)),
//           // Number of B-spline coefficients of the variable
//           std::tuple(iganet::utils::to_array(30_i64, 30_i64)));

//   // Set maximum number of epochs
//   net.options().max_epoch(500);

//   // Set tolerance for the loss functions
//   net.options().min_loss(1e-8);

//   // Start time measurement
//   auto t1 = std::chrono::high_resolution_clock::now();

//   // Train network
//   net.train(*train_loader);

//    // Stop time measurement
//   auto t2 = std::chrono::high_resolution_clock::now();
//   iganet::Log(iganet::log::info)
//       << "Training took "
//         << std::chrono::duration_cast<std::chrono::duration<double>>(t2 -
//                                                                          t1)
//               .count()
//       << " seconds\n";

//   // Save trained model to file
//   net.save(IGANET_DATA_DIR "trained/2d_quadCircle_variableRadius");

// #ifdef IGANET_WITH_GISMO
//         // Convert B-spline objects to G+Smo
//         auto G_gismo = net.G().to_gismo();
//         auto u_gismo = net.u().to_gismo();
//         auto f_gismo = net.f().to_gismo();

//         // Set up expression assembler
//         gsExprAssembler<real_t> A(1, 1);
//         gsMultiBasis<real_t> basis(u_gismo, true);

//         A.setIntegrationElements(basis);

//         auto G = A.getMap(G_gismo);
//         auto u = A.getCoeff(u_gismo, G);
//         auto f = A.getCoeff(f_gismo, G);

//         // Compute L2- and H2-error
//         gsExprEvaluator<real_t> ev(A);

//         iganet::Log(iganet::log::info)
//             << "L2-error : "
//             << gismo::math::sqrt(ev.integral((u - f).sqNorm() * meas(G)))
//             << std::endl;

//         iganet::Log(iganet::log::info)
//             << "H1-error : "
//             << gismo::math::sqrt(ev.integral(
//                    (gismo::expr::igrad(u, G) - gismo::expr::igrad(f, G))
//                        .sqNorm() *
//                    meas(G)))
//             << std::endl;
// #endif


// // evaluate model for one specific geometry
// // Load XML file
// pugi::xml_document xml;
// xml.load_file(IGANET_DATA_DIR "surfaces/2d_quadCircle_variableRadius/quadCircleR075I02_resultR1E1Fixed.xml");
// net.G().from_xml(xml);

// net.eval();

// #ifdef IGANET_WITH_MATPLOT
//   // Evaluate position of collocation points in physical domain
//   auto colPts = net.G().eval(net.collPts().first);

//   // Plot the solution
//   net.G()
//       .plot(net.u(), json)
//       ->show();

//   // Plot the difference between the solution and the reference data
//   net.G()
//       .plot(net.u().abs_diff(net.f()),
//            json)
//       ->show();
// #endif

//   // Load XML file
// xml.load_file(IGANET_DATA_DIR "surfaces/2d_quadCircle_variableRadius/quadCircleR125I04_resultR1E1Fixed.xml");
// net.G().from_xml(xml);

// net.eval();

// #ifdef IGANET_WITH_MATPLOT
//   // Evaluate position of collocation points in physical domain

//   // Plot the solution
//   net.G()
//       .plot(net.u(), json)
//       ->show();

//   // Plot the difference between the solution and the reference data
//   net.G()
//       .plot(net.u().abs_diff(net.f()),
//            json)
//       ->show();
// #endif

  return 0;
 }
