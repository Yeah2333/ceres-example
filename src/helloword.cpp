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
struct NumericDiffCostFunctor {
    bool operator()(const double* const x, double* residual) const{
        residual[0] = 10.0 - x[0];
        return true;
    }
};
// A CostFunction implementing analytically derivatives for the
// function f(x) = 10 - x.
class QuardraticCostFunction:public ceres::SizedCostFunction<1 /* number of residuals */,
        1 /* size of first parameter */ >{
public:
    virtual ~QuardraticCostFunction(){}

    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const{
        double x = parameters[0][0];
        // f(x) = 10 - x.
        residuals[0] = 10.0 - x;

        // f'(x) = -1. Since there's only 1 parameter and that parameter
        // has 1 dimension, there is only 1 element to fill in the
        // jacobians.
        //
        // Since the Evaluate function can be called with the jacobians
        // pointer equal to NULL, the Evaluate function must check to see
        // if jacobians need to be computed.
        //
        // For this simple problem it is overkill to check if jacobians[0]
        // is NULL, but in general when writing more complex
        // CostFunctions, it is possible that Ceres may only demand the
        // derivatives w.r.t. a subset of the parameter blocks.
        if (jacobians!= nullptr && jacobians[0] != nullptr) {
            jacobians[0][0] = -1;
        }
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
    ///auto-differentiation to obtain the derivative(jacobian).
    //ceres::CostFunction* cost_function =
    //        new ceres::AutoDiffCostFunction<CostFunctor,1,1>(new CostFunctor);
    //problem.AddResidualBlock(cost_function, nullptr, &x);
    ///NumericDifferentitation
//    ceres::CostFunction* costFunction = new ceres::NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL, 1, 1>(
//            new NumericDiffCostFunctor
//            );
//    problem.AddResidualBlock(costFunction, nullptr, &x);
    /// Analytic function
    ceres::CostFunction* costFunction = new QuardraticCostFunction;
    problem.AddResidualBlock(costFunction,NULL, &x);

    //Run the solver!
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, & summary);

    std::cout << summary.BriefReport() << std::endl;
    std::cout << "x" << inital_x << "->" << x << std::endl;
}

