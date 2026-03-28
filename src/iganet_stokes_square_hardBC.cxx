#include <filesystem>
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
#include <sstream>


using namespace iganet::literals;

int64_t N = 10;      // number of variable coeffs

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

  /// @brief File stream for logging loss components
  std::ofstream loss_log_file_;

  /// @brief Iteration counter
  int64_t iteration_counter_;

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
         std::vector<std::vector<std::any>> &&activations,
         const std::string& loss_log_filepath,
         Args &&...args)
      : Base(std::forward<std::vector<int64_t>>(layers),
             std::forward<std::vector<std::vector<std::any>>>(activations),
             std::forward<Args>(args)...),
        ref_(iganet::utils::to_array(N, N)),
        iteration_counter_(0) {
    loss_log_file_.open(loss_log_filepath);
    if (loss_log_file_.is_open()) {
      loss_log_file_ << "iteration,loss_momentum_x,loss_momentum_y,loss_continuity,total_loss\n";
    }
  }

  /// @brief Destructor
  ~stokes() {
    if (loss_log_file_.is_open()) {
      loss_log_file_.close();
    }
  }

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
  float_t density{1e0};
  float_t viscosity{1e0};

  /// @brief Computes the loss function
  ///
  /// @param[in] outputs Output of the network
  ///
  /// @param[in] epoch Epoch number
  torch::Tensor loss(const torch::Tensor &outputs, int64_t epoch) override {

    // Cast the network output (a raw tensor) into the proper
    // function-space format, i.e. B-spline objects for the interior
    // and boundary parts that can be evaluated.
    //std::cout << "outputs: " << outputs << std::endl;

    // replace boundary coefficients for u by 0 - for 3x3 variable coeffs
    auto outputs_view = outputs.view(-1);
    // enforce hard BCs by setting boundary coeffs to 0
    int M = N + 1;  // n+1 space dimension
    // v_x space
    for (int i = 0; i < N - 1; i++)
    {
        //std::cout << "i = " << i << std::endl;
        if (i == 0)
        {
            //std::cout << i << " " << M + 1 << std::endl;
            outputs_view.index_put_({torch::indexing::Slice(i, M+1)}, 0);    
        }
        else if (i == N - 2)
        {
            int start = (i + 1) * N + i;
            int end = start + 2 + N;
            //std::cout << start << " " << end << std::endl;
            outputs_view.index_put_({torch::indexing::Slice(start, end)}, 0);
        }
        else
        {
            int start = (i + 1) * N + i;
            int end = start + 2;
            //std::cout << start << " " << end << std::endl;
            outputs_view.index_put_({torch::indexing::Slice(start, end)}, 0);
        }
    }
    // // v_y space
    int offset = N * M;

    for (int j = 0; j < M - 1; j++)
    {
        //std::cout << "j = " << j << std::endl;
        if (j == 0)
        {
            int start = j + offset;
            //std::cout << start << " " << offset + N + 1 << std::endl;
            outputs_view.index_put_({torch::indexing::Slice(start, offset+N+1)}, 0);
        }
        else if (j == M - 2)
        {
            int start = (j + 1) * (N - 1) + j + offset;
            int end = start + 2 + (N - 1);
            //std::cout << start << " " << end << std::endl;
            outputs_view.index_put_({torch::indexing::Slice(start, end)}, 0);
        }
        else
        {
            int start = (j + 1) * (N - 1) + j + offset;
            int end = start + 2;
            //std::cout << start << " " << end << std::endl;
            outputs_view.index_put_({torch::indexing::Slice(start, end)}, 0);
        }
    }

    // scale pressure coeffs to [0,0.125]
    int start = 2 * N * M;
    outputs_view.index({torch::indexing::Slice(start, torch::indexing::None)})
      .mul_(0.125).add_(0.125);

    Base::u_.from_tensor(outputs);

    auto vel = Base::u_.template clone<0, 1>();
    auto p_y = Base::u_.template clone<0, 2>();
    auto p_x = Base::u_.template clone<2, 0>();

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

    // Compute individual loss components
    auto loss_momentum_x = torch::mse_loss(res_mom_x, *f0[0]);
    auto loss_momentum_y = torch::mse_loss(res_mom_y, *f1[0]);
    auto loss_continuity = torch::mse_loss(res_cont, *f2[0]);
    auto total_loss = loss_momentum_x + loss_momentum_y + 5e2*loss_continuity;

    // Log to file
    if (loss_log_file_.is_open()) {
      loss_log_file_ << iteration_counter_ << ","
                     << loss_momentum_x.template item<double>() << ","
                     << loss_momentum_y.template item<double>() << ","
                     << loss_continuity.template item<double>() << ","
                     << total_loss.template item<double>() << "\n";
      loss_log_file_.flush();
    }
    iteration_counter_++;
          
    return total_loss;
    }
  };

int main(int argc, char* argv[]) {
      namespace fs = std::filesystem;
    // Print help if requested
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
        std::cout << "Usage: " << argv[0] << " [options]\n";
        std::cout << "Options:\n";
        std::cout << "  -output_dir <path> Output directory for CSV files\n";
        std::cout << "  -npl <int>         Number of neurons per layer (default: 50)\n";
        std::cout << "  -ngcoef <int>      Number of B-spline coefficients for geometry (default: 2)\n";
        std::cout << "  -nvarcoef <int>    Number of B-spline coefficients for variables (default: 10)\n";
        std::cout << "  -nhl <int>         Number of hidden layers (default: 3)\n";
        std::cout << "  -me <int>          Maximum number of epochs (default: 500)\n";
        std::cout << "  --help, -h         Show this help message\n";
        exit(0);
      }
    }
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

  // Parse output directory and parameters from command line arguments
  std::string output_dir = "";
  int npl = 50; // neurons per layer
  int ngcoef = 2; // geometry B-spline coefficients
  int nvarcoef = 10; // variable B-spline coefficients
  int nhl = 3; // number of hidden layers
  int me = 500; // maximum number of epochs
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-output_dir" && i + 1 < argc) {
      output_dir = argv[i + 1];
      if (!output_dir.empty() && output_dir.back() != '/' && output_dir.back() != '\\') output_dir += "/";
      // Create output_dir if it does not exist
      fs::path outdir_path(output_dir);
      if (!output_dir.empty() && !fs::exists(outdir_path)) {
        fs::create_directories(outdir_path);
      }
      ++i;
    } else if (arg == "-npl" && i + 1 < argc) {
      npl = std::stoi(argv[i + 1]);
      ++i;
    } else if (arg == "-ngcoef" && i + 1 < argc) {
      ngcoef = std::stoi(argv[i + 1]);
      ++i;
    } else if (arg == "-nvarcoef" && i + 1 < argc) {
      nvarcoef = std::stoi(argv[i + 1]);
      ++i;
    } else if (arg == "-nhl" && i + 1 < argc) {
      nhl = std::stoi(argv[i + 1]);
      ++i;
    } else if (arg == "-me" && i + 1 < argc) {
      me = std::stoi(argv[i + 1]);
      ++i;
    }
  }

  // Define savedir as output_dir + formatted parameters
  std::stringstream savedir_ss;
  savedir_ss << output_dir << "/g" << ngcoef << "v" << nvarcoef << "nhl" << nhl << "npl" << npl;
  std::string savedir = savedir_ss.str();
  // Create savedir if it does not exist
  fs::path savedir_path(savedir);
  if (!savedir.empty() && !fs::exists(savedir_path)) {
      fs::create_directories(savedir_path);
  }

  N = nvarcoef;

  // Set up layers and activations vectors
  std::vector<int64_t> layers;
  for (int i = 0; i < nhl; ++i) {
    layers.push_back(npl);
  }
  std::vector<std::vector<std::any>> activations;
  for (int i = 0; i < nhl; ++i) {
    activations.push_back({iganet::activation::tanh});
  }
    activations.push_back({iganet::activation::none});

    // Create loss log file path
    std::string loss_log_filepath = savedir + "/loss_components.csv";

    stokes<optimizer_t, geometry_t, variable_t>
      net(
        std::move(layers),
        std::move(activations),
        loss_log_filepath,
        iganet::utils::to_array(int64_t(ngcoef), int64_t(ngcoef)),
        iganet::utils::to_array(int64_t(nvarcoef), int64_t(nvarcoef)));

    iganet::Log(iganet::log::info)
              << ", #parameters: " << net.nparameters() << std::endl;
    iganet::Log(iganet::log::info)
              << ", #parameters: " << net.nparameters() << std::endl;
    iganet::Log(iganet::log::info)
              << ", #ngcoef: " << ngcoef << std::endl;
    iganet::Log(iganet::log::info)
              << ", #nvarcoef: " << ngcoef << std::endl;
    iganet::Log(iganet::log::info)
              << ", #nhl: " << nhl << std::endl;
    iganet::Log(iganet::log::info)
              << ", #npl: " << npl << std::endl;
    iganet::Log(iganet::log::info)
              << ", #ngcoef: " << net.options().max_epoch() << std::endl;

  // prescribe body force by modifying sub-spaces of f
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

 // Set maximum number of epochs
            net.options().max_epoch(
              iganet::utils::getenv("IGANET_MAX_EPOCH", int64_t(me)));

          net.options().min_loss_rel_change(
            iganet::utils::getenv("IGANET_MIN_LOSS_REL_CHANGE", 1e-8));
          net.options().min_loss_change(
            iganet::utils::getenv("IGANET_MIN_LOSS_CHANGE", 0.0));

          // Set tolerance for the loss functions
          net.options().min_loss(
              iganet::utils::getenv("IGANET_MIN_LOSS", 1e-14));


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
  // Export sampled vx data on a regular grid for Python/matplotlib (using TensorArray input)
    {
      std::ofstream vx_grid_file(savedir + "/vx_grid_for_python.csv");
      vx_grid_file << "xi,eta,vx\n";
      int res0 = 100, res1 = 100;
      if (json.contains("res0")) res0 = json["res0"].get<int>();
      if (json.contains("res1")) res1 = json["res1"].get<int>();
      for (int i = 0; i <= res0; ++i) {
        double xi = double(i) / res0;
        for (int j = 0; j <= res1; ++j) {
          double eta = double(j) / res1;
          iganet::utils::TensorArray<2> xi_tensor = {torch::tensor({xi}), torch::tensor({eta})};
          auto val = vx.eval(xi_tensor); 
          double vx_val = val[0]->item<double>();
          vx_grid_file << xi << "," << eta << "," << vx_val << "\n";
        }
      }
      vx_grid_file.close();
    }

  auto& vy = net.u().template space<1>();
  //net.G().space().plot(vy, json)->show();

  // Export sampled vy data on a regular grid for Python/matplotlib (using TensorArray input)
    {
      std::ofstream vy_grid_file(savedir + "/vy_grid_for_python.csv");
      vy_grid_file << "xi,eta,vy\n";
      int res0 = 100, res1 = 100;
      if (json.contains("res0")) res0 = json["res0"].get<int>();
      if (json.contains("res1")) res1 = json["res1"].get<int>();
      for (int i = 0; i <= res0; ++i) {
        double xi = double(i) / res0;
        for (int j = 0; j <= res1; ++j) {
          double eta = double(j) / res1;
          iganet::utils::TensorArray<2> xi_tensor = {torch::tensor({xi}), torch::tensor({eta})};
          auto val = vy.eval(xi_tensor); 
          double vy_val = val[0]->item<double>();
          vy_grid_file << xi << "," << eta << "," << vy_val << "\n";
        }
      }
      vy_grid_file.close();
    }
  
  auto& p = net.u().template space<2>();
  //net.G().space().plot(p, json)->show();

  // Export sampled vy data on a regular grid for Python/matplotlib (using TensorArray input)
    {
      std::ofstream p_grid_file(savedir + "/p_grid_for_python.csv");
      p_grid_file << "xi,eta,vy\n";
      int res0 = 100, res1 = 100;
      if (json.contains("res0")) res0 = json["res0"].get<int>();
      if (json.contains("res1")) res1 = json["res1"].get<int>();
      for (int i = 0; i <= res0; ++i) {
        double xi = double(i) / res0;
        for (int j = 0; j <= res1; ++j) {
          double eta = double(j) / res1;
          iganet::utils::TensorArray<2> xi_tensor = {torch::tensor({xi}), torch::tensor({eta})};
          auto val = p.eval(xi_tensor); 
          double p_val = val[0]->item<double>();
          p_grid_file << xi << "," << eta << "," << p_val << "\n";
        }
      }
      p_grid_file.close();
    }


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
   net.G().space().plot(ref_vx.abs_diff(vx),  json)->show();
   net.G().space().plot(ref_vy.abs_diff(vy),  json)->show();
  // net.G().space().plot(ref_p.abs_diff(p),  json)->show();

  // net.G().space().plot(err_p_spl,  json)->show();

  // Export error fields on a regular grid for Python/matplotlib
  {
    std::ofstream err_vx_grid_file(savedir + "/err_vx_grid_for_python.csv");
    err_vx_grid_file << "xi,eta,err_vx\n";
    std::ofstream err_vy_grid_file(savedir + "/err_vy_grid_for_python.csv");
    err_vy_grid_file << "xi,eta,err_vy\n";
    std::ofstream err_p_grid_file(savedir + "/err_p_grid_for_python.csv");
    err_p_grid_file << "xi,eta,err_p\n";
    std::ofstream err_p_spl_grid_file("/err_p_spl_grid_for_python.csv");
    err_p_spl_grid_file << "xi,eta,err_p_spl\n";
    int res0 = 100, res1 = 100;
    if (json.contains("res0")) res0 = json["res0"].get<int>();
    if (json.contains("res1")) res1 = json["res1"].get<int>();
    for (int i = 0; i <= res0; ++i) {
      double xi = double(i) / res0;
      for (int j = 0; j <= res1; ++j) {
        double eta = double(j) / res1;
        iganet::utils::TensorArray<2> xi_tensor = {torch::tensor({xi}), torch::tensor({eta})};
        // Error vx
        auto err_vx_val = ref_vx.abs_diff(vx).eval(xi_tensor);
        err_vx_grid_file << xi << "," << eta << "," << err_vx_val[0]->item<double>() << "\n";
        // Error vy
        auto err_vy_val = ref_vy.abs_diff(vy).eval(xi_tensor);
        err_vy_grid_file << xi << "," << eta << "," << err_vy_val[0]->item<double>() << "\n";
        // Error p
        auto err_p_val = ref_p.abs_diff(p).eval(xi_tensor);
        err_p_grid_file << xi << "," << eta << "," << err_p_val[0]->item<double>() << "\n";
        // Error p_spl
        auto err_p_spl_val = err_p_spl.eval(xi_tensor);
        err_p_spl_grid_file << xi << "," << eta << "," << err_p_spl_val[0]->item<double>() << "\n";
      }
    }
    err_vx_grid_file.close();
    err_vy_grid_file.close();
    err_p_grid_file.close();
    err_p_spl_grid_file.close();
  }

#endif

// Export all values from first (TensorArray<2>) and second (tuple<torch::Tensor, torch::Tensor>) to CSV
{
  std::ofstream file(savedir + "/collPts_vx.csv");
  const auto& first = std::get<0>(net.collPts().first); // TensorArray<2>
  // Get sizes
  // Export as coordinate pairs: xi, eta
  // Try to export first[0] and first[1] as x and y (or xi, eta)
  auto n = first[0].size(0);
  file << "xi,eta\n";
  for (int64_t i = 0; i < n; ++i) {
    file << first[0][i].item<double>() << "," << first[1][i].item<double>() << "\n";
  }
  file.close();
}
{
  std::ofstream file(savedir + "/collPts_vy.csv");
  const auto& first = std::get<1>(net.collPts().first); // TensorArray<2>
  // Get sizes
  // Export as coordinate pairs: xi, eta
  // Try to export first[0] and first[1] as x and y (or xi, eta)
  auto n = first[0].size(0);
  file << "xi,eta\n";
  for (int64_t i = 0; i < n; ++i) {
    file << first[0][i].item<double>() << "," << first[1][i].item<double>() << "\n";
  }
  file.close();
}
{
  std::ofstream file(savedir + "/collPts_p.csv");
  const auto& first = std::get<2>(net.collPts().first); // TensorArray<2>
  // Get sizes
  // Export as coordinate pairs: xi, eta
  // Try to export first[0] and first[1] as x and y (or xi, eta)
  auto n = first[0].size(0);
  file << "xi,eta\n";
  for (int64_t i = 0; i < n; ++i) {
    file << first[0][i].item<double>() << "," << first[1][i].item<double>() << "\n";
  }
  file.close();
}
return 0;
}


