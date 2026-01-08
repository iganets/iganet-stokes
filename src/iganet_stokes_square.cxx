/**
   @file examples/iganet_stokes.cxx

   @brief Demonstration of IgANet Stokes solver

   This example demonstrates how to implement a simple IgANet for
   learning the Stokes equation with (non-)homogeneous Dirichlet
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

/// @brief Specialization of the abstract IgANet class for Stokes's equation
template <typename Optimizer, typename GeometryMap, typename Variable>
class stokes : public iganet::v1::IgANet<Optimizer, GeometryMap, Variable>,
               public iganet::v1::IgANetCustomizable<GeometryMap, Variable> {

private:
  /// @brief Type of the base class
  using Base = iganet::v1::IgANet<Optimizer, GeometryMap, Variable>;

  /// @brief Collocation points
  typename Base::variable_collPts_type collPts_;

  /// @brief Reference solution
  Variable ref_;

  /// @brief Type of the customizable class
  using Customizable = iganet::v1::IgANetCustomizable<GeometryMap, Variable>;

  /// @brief Knot indices of variables
  typename Customizable::variable_interior_knot_indices_type var_knot_indices_;

  /// @brief Coefficient indices of variables
  typename Customizable::variable_interior_coeff_indices_type
      var_coeff_indices_;


public:
  /// @brief Constructor
  template <typename... Args>
  stokes(std::vector<int64_t> &&layers,
         std::vector<std::vector<std::any>> &&activations, Args &&...args)
      : Base(std::forward<std::vector<int64_t>>(layers),
             std::forward<std::vector<std::vector<std::any>>>(activations),
             std::forward<Args>(args)...),
        ref_(iganet::utils::to_array(10_i64, 10_i64)) {}

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
    // In the very first epoch we need to generate the sampling points
    // for the inputs and the sampling points in the function space of
    // the variables since otherwise the respective tensors would be
    // empty. In all further epochs no updates are needed since we do
    // not change the inputs nor the variable function space.
    if (epoch == 0) {
      Base::inputs(epoch);
      collPts_ = Base::variable_collPts(iganet::collPts::greville);

      var_knot_indices_ =
          Base::u_.template find_knot_indices<iganet::functionspace::interior>(
              collPts_.first);
      var_coeff_indices_ =
          Base::u_.template find_coeff_indices<iganet::functionspace::interior>(
              var_knot_indices_);

      return true;
    } else
      return false;
  }
  // define material parameters
  float_t density{1e3};
  float_t viscosity{1e-3};

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
    std::cout << "outputs: " << outputs << std::endl;

    auto vel = Base::u_.template clone<0, 1>();
    auto p_y = Base::u_.template clone<0, 2>();
    auto p_x = Base::u_.template clone<2, 0>();

    // scale pressure
    //auto p_y_scaled_coeffs = 0.125*p_y.template coeffs()[0]+0.125;
    //p_y.from_tensor(p_y_scaled_coeffs);
    //auto p_x_scaled_coeffs = 0.125*p_x.template coeffs()[0]+0.125;
    //p_x.from_tensor(p_x_scaled_coeffs);
    
    // Compute first derivatives
    auto vel_grad= vel.grad( std::get<2>(collPts_.first) ); // du/dx [ cpt_p ] 
    //std::cout << ", u_grad_x_mass: " << *vel_grad[0] << std::endl; //du/dx 
    
    auto p_grad_mom_x = p_x.grad( std::get<0>(collPts_.first) )[0]; // dp [cpt_u]
    auto p_grad_mom_y = p_y.grad(std::get<1>(collPts_.first) )[1]; //dp [cpt_v]
    //std::cout << ", p_grad_mom_x: " << *p_grad_mom_x << std::endl; //dp/dx
    //std::cout << ", p_grad_mom_y: " << *p_grad_mom_y << std::endl; //dp/dy

    // Compute second derivatives
    auto vel_hess_mom_x = vel.hess( std::get<0>(collPts_.first) ); //  [cpt_u ] 
    //std::cout << "d2u_dxx: " << *vel_hess_mom_x[0] << std::endl; // u_xx
    //std::cout << "d2u_dyy: " << *vel_hess_mom_x[3] << std::endl; // u_yy
    auto vel_hess_mom_y = vel.hess(std::get<1>(collPts_.first) ); //  [cpt_v]
    //std::cout << "d2v_dxx: " << *vel_hess_mom_y[4] << std::endl; // v_xx
    //std::cout << "d2v_dyy: " << *vel_hess_mom_y[7] << std::endl; // v_yy

    // body force
    auto f = Base::f_.eval(collPts_.first);
    auto f0 = std::get<0>(f);
    auto f1 = std::get<1>(f);
    auto f2 = std::get<2>(f);

    // loss
    auto res_mom_x = *p_grad_mom_x - (*vel_hess_mom_x[0]+*vel_hess_mom_x[3]);
    auto res_mom_y = *p_grad_mom_y - (*vel_hess_mom_y[4]+*vel_hess_mom_y[7]);
    auto res_cont = *vel_grad[0] + *vel_grad[1];

    auto sol_bdr = Base::u_.template eval<iganet::functionspace::boundary>(
        collPts_.second);
    auto sol_bdrx = std::get<0>(sol_bdr);
    auto sol_bdry = std::get<1>(sol_bdr);
    auto sol_bdrp = std::get<2>(sol_bdr);

    auto bdr =
        ref_.template eval<iganet::functionspace::boundary>(collPts_.second);
       
    auto bdr_vx = std::get<0>(bdr);
    auto bdr_vy = std::get<1>(bdr);
    auto bdr_p = std::get<2>(bdr);
    
    //std::cout << "bdr: " << bdr << std::endl;
          
    return torch::mse_loss(res_mom_x, *f0[0]) +
            torch::mse_loss(res_mom_y, *f1[0]) +
            5e1*torch::mse_loss(res_cont, *f2[0]) +
            1e1*torch::mse_loss(*std::get<0>(sol_bdrx)[0], *std::get<0>(bdr_vx)[0]) +
            1e1*torch::mse_loss(*std::get<1>(sol_bdrx)[0], *std::get<1>(bdr_vx)[0]) +
            1e1*torch::mse_loss(*std::get<2>(sol_bdrx)[0], *std::get<2>(bdr_vx)[0]) +
            1e1*torch::mse_loss(*std::get<3>(sol_bdrx)[0], *std::get<3>(bdr_vx)[0]) +
            1e1*torch::mse_loss(*std::get<0>(sol_bdry)[0], *std::get<0>(bdr_vy)[0]) +
            1e1*torch::mse_loss(*std::get<1>(sol_bdry)[0], *std::get<1>(bdr_vy)[0]) +
            1e1*torch::mse_loss(*std::get<2>(sol_bdry)[0], *std::get<2>(bdr_vy)[0]) +
            1e1*torch::mse_loss(*std::get<3>(sol_bdry)[0], *std::get<3>(bdr_vy)[0]);
    }
  };

int main() {
  iganet::init();
  //iganet::init(iganet::Log(iganet::log::verbose));
  iganet::Log.setLogLevel(iganet::log::verbose);

  nlohmann::json json;
  json["res0"] = 50;
  json["res1"] = 50;
  json["cnet"] = true;

  using namespace iganet::literals;
  using optimizer_t = torch::optim::LBFGS;
  using real_t = double;

  using geometry_t = iganet::S<iganet::UniformBSpline<real_t, 2, 1, 1>>;
  //using variable_t = iganet::TH<iganet::NonUniformBSpline<real_t, 1, 1, 1>,2>;
  using variable_t = iganet::RT<iganet::UniformBSpline<real_t, 1, 2, 2>,2>;

  stokes<optimizer_t, geometry_t, variable_t>
      net( // Number of neurons per layers
          {50, 50, 50},
          // Activation functions
          {{iganet::activation::tanh},
          {iganet::activation::tanh},
          {iganet::activation::tanh},
           {iganet::activation::none}},
          // Number of B-spline coefficients of the geometry, just [0,1] x [0,1]
          iganet::utils::to_array(2_i64, 2_i64),
          // Number of B-spline coefficients of the variable
          iganet::utils::to_array(10_i64, 10_i64));

    iganet::Log(iganet::log::info)
              << ", #parameters: " << net.nparameters() << std::endl;

 
  // prescribe boundary force by modifying sub-spaces of f
  auto& f0 = net.f().template space<0>();
  f0.transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 1>{
      (12-24*xi[1])*xi[0]*xi[0]*xi[0]*xi[0] +(-24+48*xi[1])*xi[0]*xi[0]*xi[0]+(-48*xi[1]+72*xi[1]*xi[1]-48*xi[1]*xi[1]*xi[1]+12)*xi[0]*xi[0]+(-2+24*xi[1]-72*xi[1]*xi[1]+48*xi[1]*xi[1]*xi[1])*xi[0]+1-4*xi[1]+12*xi[1]*xi[1]-8*xi[1]*xi[1]*xi[1]
    };
  });

  auto& f1 = net.f().template space<1>();
  f1.transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 1>{
      (8-48*xi[1]+48*xi[1]*xi[1])*xi[0]*xi[0]*xi[0]+(-12+72*xi[1]-72*xi[1]*xi[1])*xi[0]*xi[0]+(4-24*xi[1]+48*xi[1]*xi[1]-48*xi[1]*xi[1]*xi[1]+24*xi[1]*xi[1]*xi[1]*xi[1])*xi[0]-12*xi[1]*xi[1]+24*xi[1]*xi[1]*xi[1]-12*xi[1]*xi[1]*xi[1]*xi[1]
    };
  });

  //f2 is 0.0 by default

  // impose boundary conditions
  net.ref().template boundary<0>().template side<iganet::north>().transform(
    [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0};
      });
  net.ref().template boundary<1>().template side<iganet::north>().transform(
    [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0};
      });

  net.ref().template boundary<0>().template side<iganet::south>().transform(
    [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0};
      });
  net.ref().template boundary<1>().template side<iganet::south>().transform(
    [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0};
      });
  
  net.ref().template boundary<0>().template side<iganet::east>().transform(
    [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0};
      });
  net.ref().template boundary<1>().template side<iganet::east>().transform(
    [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0};
      });

  net.ref().template boundary<0>().template side<iganet::west>().transform(
    [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0};
      });
  net.ref().template boundary<1>().template side<iganet::west>().transform(
    [](const std::array<real_t, 1> xi) {
        return std::array<real_t, 1>{0.0};
      });

 // Set maximum number of epochs
          net.options().max_epoch(
              iganet::utils::getenv("IGANET_MAX_EPOCH", 1_i64));

          // Set tolerance for the loss functions
          net.options().min_loss(
              iganet::utils::getenv("IGANET_MIN_LOSS", 1e-12));


  // Start time measurement
  auto t1 = std::chrono::high_resolution_clock::now();

  // Train network
  net.train();

  // Stop time measurement
  auto t2 = std::chrono::high_resolution_clock::now();
  //iganet::Log(iganet::log::info) << "net.f0: " << net.f()[0] << std::endl; 

  iganet::Log(iganet::log::info)
      << "Training took "
      << std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
             .count()
      << " seconds\n";

// Compute analytical solution
  auto& ref_vx = net.ref().template space<0>();
  ref_vx.transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 1>{
      xi[0]*xi[0]*(1.0-xi[0])*(1.0-xi[0])*(2.0*xi[1]-6.0*xi[1]*xi[1]+4.0*xi[1]*xi[1]*xi[1])
    };
  });

  auto& ref_vy = net.ref().template space<1>();
  ref_vy.transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 1>{
      -xi[1]*xi[1]*(1.0-xi[1])*(1.0-xi[1])*(2.0*xi[0]-6.0*xi[0]*xi[0]+4.0*xi[0]*xi[0]*xi[0])
    };
  });

  auto& ref_p = net.ref().template space<2>();
  ref_p.transform([](const std::array<real_t, 2> xi) {
    return std::array<real_t, 1>{
      xi[0]*(1.0-xi[0])
    };
  });

#ifdef IGANET_WITH_MATPLOT
  // Plot the solution
  // get solution components
  auto& vx = net.u().template space<0>();
  //net.G().space().plot(vx, json)->show();

  auto& vy = net.u().template space<1>();
  //net.G().space().plot(vy, json)->show();

  auto& p = net.u().template space<2>();
  //net.G().space().plot(p, json)->show();

  auto min_ref_p = torch::min(ref_p.coeffs()[0]);
  auto max_ref_p = torch::max(ref_p.coeffs()[0]);
  auto range_p_ref = max_ref_p - min_ref_p;
  //std::cout << "min ref_p: " << min_ref_p << std::endl;
  //std::cout << "max ref_p: " << max_ref_p << std::endl;
  //std::cout << "range ref_p: " << range_p_ref << std::endl;

  auto max_pred_p = torch::max(p.coeffs()[0]);
  //std::cout << "max pred_p: " << max_pred_p << std::endl;

  // compute pressure error
  auto err_p = abs((p.coeffs()[0] - max_pred_p) - (ref_p.coeffs()[0] - max_ref_p)/range_p_ref);
  // cast error to function space
  auto err_p_spl = p.clone();
  err_p_spl.from_tensor(err_p);
  
  // Plot the difference between the exact and predicted solutions
  //net.G().space().plot(ref_vx.abs_diff(vx),  json)->show();
  //net.G().space().plot(ref_vy.abs_diff(vy),  json)->show();
  //net.G().space().plot(err_p_spl,  json)->show();

#endif

  iganet::finalize();
  return 0;
}
