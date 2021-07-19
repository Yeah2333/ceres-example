#include <iostream>
#include <ceres/ceres.h>
#include <ceres/problem.h>

struct CostFunctor {
    template<class T>
    bool operator()(const T* const x, T* residual) const{
        residual[0] = 10.0 - x[0];
        return true;
    }
};
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    //This variable to solve for with its inital value.
    double inital_x = 5.0;
    double x = inital_x;

    //Build the problem
    ceres::Problem problem;

    //Set up the only cost function(also known as residual). This uses
    //auto-differentiation to obtain the derivative(jacobian).
    ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<CostFunctor,1,1>(new CostFunctor);
    problem.AddResidualBlock(cost_function, nullptr, &x);

    //Run the solver!
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, & summary);

    std::cout << summary.BriefReport() << std::endl;
    std::cout << "x" << inital_x << "->" << x << std::endl;
}

