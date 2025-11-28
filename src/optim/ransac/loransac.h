//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_OPTIM_RANSAC_LORANSAC_H_
#define SENSEMAP_OPTIM_RANSAC_LORANSAC_H_

#include <cfloat>
#include <random>
#include <stdexcept>
#include <vector>

#include "random_sampler.h"
#include "ransac.h"
#include "support_measurement.h"
#include "util/logging.h"
#include "util/alignment.h"
namespace sensemap {

// Implementation of LO-RANSAC (Locally Optimized RANSAC).
//
// "Locally Optimized RANSAC" Ondrej Chum, Jiri Matas, Josef Kittler, DAGM 2003.
template <typename Estimator, typename LocalEstimator,
          typename SupportMeasurer = InlierSupportMeasurer,
          typename Sampler = RandomSampler>
class LORANSAC : public RANSAC<Estimator, SupportMeasurer, Sampler> {
public:
    using typename RANSAC<Estimator, SupportMeasurer, Sampler>::Report;

    explicit LORANSAC(const RANSACOptions& options);

    // Robustly estimate model with RANSAC (RANdom SAmple Consensus).
    //
    // @param X              Independent variables.
    // @param Y              Dependent variables.
    //
    // @return               The report with the results of the estimation.
    Report Estimate(const std::vector<typename Estimator::X_t>& X,
                    const std::vector<typename Estimator::Y_t>& Y);

    Report EstimateMultiple(const std::vector<typename Estimator::X_t>& X,
                            const std::vector<typename Estimator::Y_t>& Y);

    Report Estimate(const std::vector<typename Estimator::X_t>& X);

    Report EstimateMultiple(const std::vector<typename Estimator::X_t>& X);

    // Objects used in RANSAC procedure.
    using RANSAC<Estimator, SupportMeasurer, Sampler>::estimator;
    LocalEstimator local_estimator;
    using RANSAC<Estimator, SupportMeasurer, Sampler>::sampler;
    using RANSAC<Estimator, SupportMeasurer, Sampler>::support_measurer;

private:
    using RANSAC<Estimator, SupportMeasurer, Sampler>::options_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

template <typename Estimator, typename LocalEstimator, typename SupportMeasurer,
          typename Sampler>
LORANSAC<Estimator, LocalEstimator, SupportMeasurer, Sampler>::LORANSAC(
    const RANSACOptions& options)
    : RANSAC<Estimator, SupportMeasurer, Sampler>(options) {}

template <typename Estimator, typename LocalEstimator, typename SupportMeasurer,
          typename Sampler>
typename LORANSAC<Estimator, LocalEstimator, SupportMeasurer, Sampler>::Report
LORANSAC<Estimator, LocalEstimator, SupportMeasurer, Sampler>::Estimate(
    const std::vector<typename Estimator::X_t>& X,
    const std::vector<typename Estimator::Y_t>& Y) {
    CHECK_EQ(X.size(), Y.size());

    const size_t num_samples = X.size();

    typename RANSAC<Estimator, SupportMeasurer, Sampler>::Report report;
    report.success = false;
    report.num_trials = 0;

    if (num_samples < Estimator::kMinNumSamples) {
        return report;
    }

    typename SupportMeasurer::Support best_support;
    typename Estimator::M_t best_model;
    bool best_model_is_local = false;

    bool abort = false;

    const double max_residual = options_.max_error * options_.max_error;

    std::vector<double> residuals(num_samples);

    std::vector<typename LocalEstimator::X_t> X_inlier;
    std::vector<typename LocalEstimator::Y_t> Y_inlier;

    std::vector<typename Estimator::X_t> X_rand(Estimator::kMinNumSamples);
    std::vector<typename Estimator::Y_t> Y_rand(Estimator::kMinNumSamples);

    sampler.Initialize(num_samples);

    size_t max_num_trials = options_.max_num_trials;
    max_num_trials = std::min<size_t>(max_num_trials, sampler.MaxNumSamples());
    size_t dyn_max_num_trials = max_num_trials;

    for (report.num_trials = 0; report.num_trials < max_num_trials; ++report.num_trials) {
        if (abort) {
            report.num_trials += 1;
            break;
        }

        sampler.SampleXY(X, Y, &X_rand, &Y_rand);

        // Estimate model for current subset.
        const std::vector<typename Estimator::M_t> sample_models =
            estimator.Estimate(X_rand, Y_rand);

        // Iterate through all estimated models
        for (const auto& sample_model : sample_models) {
            estimator.Residuals(X, Y, sample_model, &residuals);
            CHECK_EQ(residuals.size(), X.size());

            const auto support = support_measurer.Evaluate(residuals, max_residual);

            // Do local optimization if better than all previous subsets.
            if (support_measurer.Compare(support, best_support)) {
                best_support = support;
                best_model = sample_model;
                best_model_is_local = false;

                // Estimate locally optimized model from inliers.
                if (support.num_inliers > Estimator::kMinNumSamples &&
                    support.num_inliers >= LocalEstimator::kMinNumSamples) {
                    X_inlier.clear();
                    Y_inlier.clear();
                    X_inlier.reserve(support.num_inliers);
                    Y_inlier.reserve(support.num_inliers);
                    for (size_t i = 0; i < residuals.size(); ++i) {
                        if (residuals[i] <= max_residual) {
                            X_inlier.push_back(X[i]);
                            Y_inlier.push_back(Y[i]);
                        }
                    }

                    const std::vector<typename LocalEstimator::M_t> local_models =
                        local_estimator.Estimate(X_inlier, Y_inlier);

                    for (const auto& local_model : local_models) {
                        local_estimator.Residuals(X, Y, local_model, &residuals);
                        CHECK_EQ(residuals.size(), X.size());

                        const auto local_support =
                            support_measurer.Evaluate(residuals, max_residual);

                        // Check if non-locally optimized model is better.
                        if (support_measurer.Compare(local_support, best_support)) {
                          best_support = local_support;
                          best_model = local_model;
                          best_model_is_local = true;
                        }
                    }
                }

                dyn_max_num_trials =
                    RANSAC<Estimator, SupportMeasurer, Sampler>::ComputeNumTrials(
                        best_support.num_inliers, num_samples, options_.confidence);
            }

            if (report.num_trials >= dyn_max_num_trials &&
                report.num_trials >= options_.min_num_trials) {
                abort = true;
                break;
            }
        }
    }

    report.support = best_support;
    report.model = best_model;

    // No valid model was found
    if (report.support.num_inliers < estimator.kMinNumSamples) {
        return report;
    }

    report.success = true;

    // Determine inlier mask. Note that this calculates the residuals for the
    // best model twice, but saves to copy and fill the inlier mask for each
    // evaluated model. Some benchmarking revealed that this approach is faster.

    if (best_model_is_local) {
        local_estimator.Residuals(X, Y, report.model, &residuals);
    } else {
        estimator.Residuals(X, Y, report.model, &residuals);
    }

    CHECK_EQ(residuals.size(), X.size());

    report.inlier_mask.resize(num_samples);
    for (size_t i = 0; i < residuals.size(); ++i) {
        if (residuals[i] <= max_residual) {
            report.inlier_mask[i] = true;
        } else {
            report.inlier_mask[i] = false;
        }
    }

    return report;
}

template <typename Estimator, typename LocalEstimator, typename SupportMeasurer,
          typename Sampler>
typename LORANSAC<Estimator, LocalEstimator, SupportMeasurer, Sampler>::Report
LORANSAC<Estimator, LocalEstimator, SupportMeasurer, Sampler>::EstimateMultiple(
    const std::vector<typename Estimator::X_t>& X,
    const std::vector<typename Estimator::Y_t>& Y) {
    CHECK_EQ(X.size(), Y.size());

    const size_t num_samples = X.size();

    typename RANSAC<Estimator, SupportMeasurer, Sampler>::Report report;
    report.success = false;
    report.num_trials = 0;

    if (num_samples < Estimator::kMinNumSamples) {
        return report;
    }

    std::vector<std::pair<size_t,typename Estimator::M_t>> all_models;
    std::vector<typename SupportMeasurer::Support> all_supports;

    typename SupportMeasurer::Support best_support;
    typename Estimator::M_t best_model;
    bool best_model_is_local = false;

    bool abort = false;

    const double max_residual = options_.max_error * options_.max_error;

    std::vector<double> residuals(num_samples);

    std::vector<typename LocalEstimator::X_t> X_inlier;
    std::vector<typename LocalEstimator::Y_t> Y_inlier;

    std::vector<typename Estimator::X_t> X_rand(Estimator::kMinNumSamples);
    std::vector<typename Estimator::Y_t> Y_rand(Estimator::kMinNumSamples);

    sampler.Initialize(num_samples);

    size_t max_num_trials = options_.max_num_trials;
    max_num_trials = std::min<size_t>(max_num_trials, sampler.MaxNumSamples());
    size_t dyn_max_num_trials = max_num_trials;

    for (report.num_trials = 0; report.num_trials < max_num_trials; ++report.num_trials) {
        if (abort) {
            report.num_trials += 1;
            break;
        }

        sampler.SampleXY(X, Y, &X_rand, &Y_rand);

        // Estimate model for current subset.
        const std::vector<typename Estimator::M_t> sample_models =
            estimator.Estimate(X_rand, Y_rand);

        // Iterate through all estimated models
        for (const auto& sample_model : sample_models) {
            //std::cout<<std::endl<<sample_model<<std::endl;

            estimator.Residuals(X, Y, sample_model, &residuals);
            CHECK_EQ(residuals.size(), X.size());

            const auto support = support_measurer.Evaluate(residuals, max_residual);
            
            all_models.emplace_back(support.num_inliers,sample_model);
            all_supports.emplace_back(support);
            
            // Do local optimization if better than all previous subsets.
            if (support_measurer.Compare(support, best_support)) {
                best_support = support;
                best_model = sample_model;
                best_model_is_local = false;

                // Estimate locally optimized model from inliers.
                if (support.num_inliers > Estimator::kMinNumSamples &&
                    support.num_inliers >= LocalEstimator::kMinNumSamples) {
                    X_inlier.clear();
                    Y_inlier.clear();
                    X_inlier.reserve(support.num_inliers);
                    Y_inlier.reserve(support.num_inliers);
                    for (size_t i = 0; i < residuals.size(); ++i) {
                        if (residuals[i] <= max_residual) {
                            X_inlier.push_back(X[i]);
                            Y_inlier.push_back(Y[i]);
                        }
                    }

                    const std::vector<typename LocalEstimator::M_t> local_models =
                        local_estimator.Estimate(X_inlier, Y_inlier);

                    for (const auto& local_model : local_models) {
                        local_estimator.Residuals(X, Y, local_model, &residuals);
                        CHECK_EQ(residuals.size(), X.size());

                        const auto local_support =
                            support_measurer.Evaluate(residuals, max_residual);

                        all_models.emplace_back(local_support.num_inliers,local_model);
                        all_supports.emplace_back(local_support);

                        // Check if non-locally optimized model is better.
                        if (support_measurer.Compare(local_support, best_support)) {
                          best_support = local_support;
                          best_model = local_model;
                          best_model_is_local = true;
                        }
                    }
                }

                dyn_max_num_trials =
                    RANSAC<Estimator, SupportMeasurer, Sampler>::ComputeNumTrials(
                        best_support.num_inliers, num_samples, options_.confidence);
            }

            if (report.num_trials >= dyn_max_num_trials &&
                report.num_trials >= options_.min_num_trials) {
                abort = true;
                break;
            }
        }
    }

    report.support = best_support;
    report.model = best_model;

    // No valid model was found
    if (report.support.num_inliers < estimator.kMinNumSamples) {
        std::cout<<"best num inliers: "<<report.support.num_inliers<<std::endl;
        return report;
    }

    report.success = true;

    // preserve all the good models
    size_t best_num_inliers=best_support.num_inliers;
    report.multiple_models.reserve(all_models.size());
    report.multiple_supports.reserve(all_supports.size());
    for (size_t i = 0; i < all_models.size(); ++i) {
        const auto& model = all_models[i];
        const auto& support = all_supports[i];
        if(static_cast<double>(model.first) >= 
                                static_cast<double>(best_num_inliers)*
                                options_.min_inlier_ratio_to_best_model){

            report.multiple_models.push_back(model.second);
            report.multiple_supports.push_back(support);
        }
    }

    // Determine inlier mask. Note that this calculates the residuals for the
    // best model twice, but saves to copy and fill the inlier mask for each
    // evaluated model. Some benchmarking revealed that this approach is faster.

    if (best_model_is_local) {
        local_estimator.Residuals(X, Y, report.model, &residuals);
    } else {
        estimator.Residuals(X, Y, report.model, &residuals);
    }

    CHECK_EQ(residuals.size(), X.size());

    report.inlier_mask.resize(num_samples);
    for (size_t i = 0; i < residuals.size(); ++i) {
        if (residuals[i] <= max_residual) {
            report.inlier_mask[i] = true;
        } else {
            report.inlier_mask[i] = false;
        }
    }

    return report;
}

template <typename Estimator, typename LocalEstimator, typename SupportMeasurer,
          typename Sampler>
typename LORANSAC<Estimator, LocalEstimator, SupportMeasurer, Sampler>::Report
LORANSAC<Estimator, LocalEstimator, SupportMeasurer, Sampler>::Estimate(const std::vector<typename Estimator::X_t>& X) {
    
    const size_t num_samples = X.size();

    typename RANSAC<Estimator, SupportMeasurer, Sampler>::Report report;
    report.success = false;
    report.num_trials = 0;

    if (num_samples < Estimator::kMinNumSamples) {
        return report;
    }

    typename SupportMeasurer::Support best_support;
    typename Estimator::M_t best_model;
    bool best_model_is_local = false;

    bool abort = false;

    const double max_residual = options_.max_error * options_.max_error;

    std::vector<double> residuals(num_samples);

    std::vector<typename LocalEstimator::X_t> X_inlier;

    std::vector<typename Estimator::X_t> X_rand(Estimator::kMinNumSamples);

    sampler.Initialize(num_samples);

    size_t max_num_trials = options_.max_num_trials;
    max_num_trials = std::min<size_t>(max_num_trials, sampler.MaxNumSamples());
    size_t dyn_max_num_trials = max_num_trials;

    for (report.num_trials = 0; report.num_trials < max_num_trials; ++report.num_trials) {
        if (abort) {
            report.num_trials += 1;
            break;
        }

        sampler.SampleX(X, &X_rand);

        // Estimate model for current subset.
        const std::vector<typename Estimator::M_t> sample_models =
            estimator.Estimate(X_rand);

        // Iterate through all estimated models
        for (const auto& sample_model : sample_models) {
            estimator.Residuals(X, sample_model, &residuals);
            CHECK_EQ(residuals.size(), X.size());

            const auto support = support_measurer.Evaluate(residuals, max_residual);

            // Do local optimization if better than all previous subsets.
            if (support_measurer.Compare(support, best_support)) {
                best_support = support;
                best_model = sample_model;
                best_model_is_local = false;

                // Estimate locally optimized model from inliers.
                if (support.num_inliers > Estimator::kMinNumSamples &&
                    support.num_inliers >= LocalEstimator::kMinNumSamples) {
                    X_inlier.clear();
                    X_inlier.reserve(support.num_inliers);
                    for (size_t i = 0; i < residuals.size(); ++i) {
                        if (residuals[i] <= max_residual) {
                            X_inlier.push_back(X[i]);
                        }
                    }

                    const std::vector<typename LocalEstimator::M_t> local_models = local_estimator.Estimate(X_inlier);

                    for (const auto& local_model : local_models) {
                        local_estimator.Residuals(X, local_model, &residuals);
                        CHECK_EQ(residuals.size(), X.size());

                        const auto local_support =
                            support_measurer.Evaluate(residuals, max_residual);

                        // Check if non-locally optimized model is better.
                        if (support_measurer.Compare(local_support, best_support)) {
                          best_support = local_support;
                          best_model = local_model;
                          best_model_is_local = true;
                        }
                    }
                }

                dyn_max_num_trials =
                    RANSAC<Estimator, SupportMeasurer, Sampler>::ComputeNumTrials(
                        best_support.num_inliers, num_samples, options_.confidence);
            }

            if (report.num_trials >= dyn_max_num_trials &&
                report.num_trials >= options_.min_num_trials) {
                abort = true;
                break;
            }
        }
    }

    report.support = best_support;
    report.model = best_model;

    // No valid model was found
    if (report.support.num_inliers < estimator.kMinNumSamples) {
        return report;
    }

    report.success = true;

    // Determine inlier mask. Note that this calculates the residuals for the
    // best model twice, but saves to copy and fill the inlier mask for each
    // evaluated model. Some benchmarking revealed that this approach is faster.

    if (best_model_is_local) {
        local_estimator.Residuals(X, report.model, &residuals);
    } else {
        estimator.Residuals(X, report.model, &residuals);
    }

    CHECK_EQ(residuals.size(), X.size());

    report.inlier_mask.resize(num_samples);
    for (size_t i = 0; i < residuals.size(); ++i) {
        if (residuals[i] <= max_residual) {
            report.inlier_mask[i] = true;
        } else {
            report.inlier_mask[i] = false;
        }
    }

    return report;
}

template <typename Estimator, typename LocalEstimator, typename SupportMeasurer,
          typename Sampler>
typename LORANSAC<Estimator, LocalEstimator, SupportMeasurer, Sampler>::Report
LORANSAC<Estimator, LocalEstimator, SupportMeasurer, Sampler>::EstimateMultiple(const std::vector<typename Estimator::X_t>& X) {
    
    const size_t num_samples = X.size();

    typename RANSAC<Estimator, SupportMeasurer, Sampler>::Report report;
    report.success = false;
    report.num_trials = 0;

    if (num_samples < Estimator::kMinNumSamples) {
        return report;
    }

    std::vector<std::pair<size_t,typename Estimator::M_t>> all_models;
    std::vector<typename SupportMeasurer::Support> all_supports;

    typename SupportMeasurer::Support best_support;
    typename Estimator::M_t best_model;
    bool best_model_is_local = false;

    bool abort = false;

    const double max_residual = options_.max_error * options_.max_error;

    std::vector<double> residuals(num_samples);

    std::vector<typename LocalEstimator::X_t> X_inlier;

    std::vector<typename Estimator::X_t> X_rand(Estimator::kMinNumSamples);

    sampler.Initialize(num_samples);

    size_t max_num_trials = options_.max_num_trials;
    max_num_trials = std::min<size_t>(max_num_trials, sampler.MaxNumSamples());
    size_t dyn_max_num_trials = max_num_trials;

    for (report.num_trials = 0; report.num_trials < max_num_trials; ++report.num_trials) {
        if (abort) {
            report.num_trials += 1;
            break;
        }

        sampler.SampleX(X, &X_rand);

        // Estimate model for current subset.
        const std::vector<typename Estimator::M_t> sample_models =
            estimator.Estimate(X_rand);

        // Iterate through all estimated models
        for (const auto& sample_model : sample_models) {
            estimator.Residuals(X, sample_model, &residuals);
            CHECK_EQ(residuals.size(), X.size());

            const auto support = support_measurer.Evaluate(residuals, max_residual);

            // Do local optimization if better than all previous subsets.
            if (support_measurer.Compare(support, best_support)) {
                best_support = support;
                best_model = sample_model;
                best_model_is_local = false;

                // Estimate locally optimized model from inliers.
                if (support.num_inliers > Estimator::kMinNumSamples &&
                    support.num_inliers >= LocalEstimator::kMinNumSamples) {
                    X_inlier.clear();
                    X_inlier.reserve(support.num_inliers);
                    for (size_t i = 0; i < residuals.size(); ++i) {
                        if (residuals[i] <= max_residual) {
                            X_inlier.push_back(X[i]);
                        }
                    }

                    const std::vector<typename LocalEstimator::M_t> local_models = local_estimator.Estimate(X_inlier);

                    for (const auto& local_model : local_models) {
                        local_estimator.Residuals(X, local_model, &residuals);
                        CHECK_EQ(residuals.size(), X.size());

                        const auto local_support =
                            support_measurer.Evaluate(residuals, max_residual);

                        // Check if non-locally optimized model is better.
                        if (support_measurer.Compare(local_support, best_support)) {
                          best_support = local_support;
                          best_model = local_model;
                          best_model_is_local = true;
                        }
                    }
                    all_models.emplace_back(best_support.num_inliers,
                                            best_model);
                    all_supports.emplace_back(best_support);
                }

                dyn_max_num_trials =
                    RANSAC<Estimator, SupportMeasurer, Sampler>::ComputeNumTrials(
                        best_support.num_inliers, num_samples, options_.confidence);
            }

            if (report.num_trials >= dyn_max_num_trials &&
                report.num_trials >= options_.min_num_trials) {
                abort = true;
                break;
            }
        }
    }

    report.support = best_support;
    report.model = best_model;

    // No valid model was found
    if (report.support.num_inliers < estimator.kMinNumSamples) {
        return report;
    }

    report.success = true;

    // preserve all the good models
    size_t best_num_inliers = best_support.num_inliers;
    report.multiple_models.reserve(all_models.size());
    report.multiple_supports.reserve(all_supports.size());
    for (size_t i = 0; i < all_models.size(); ++i) {
        const auto& model = all_models[i];
        const auto& support = all_supports[i];
        if(static_cast<double>(model.first) >= 
                                static_cast<double>(best_num_inliers)*
                                options_.min_inlier_ratio_to_best_model){

            report.multiple_models.push_back(model.second);
            report.multiple_supports.push_back(support);
        }
    }

    // Determine inlier mask. Note that this calculates the residuals for the
    // best model twice, but saves to copy and fill the inlier mask for each
    // evaluated model. Some benchmarking revealed that this approach is faster.

    if (best_model_is_local) {
        local_estimator.Residuals(X, report.model, &residuals);
    } else {
        estimator.Residuals(X, report.model, &residuals);
    }

    CHECK_EQ(residuals.size(), X.size());

    report.inlier_mask.resize(num_samples);
    for (size_t i = 0; i < residuals.size(); ++i) {
        if (residuals[i] <= max_residual) {
            report.inlier_mask[i] = true;
        } else {
            report.inlier_mask[i] = false;
        }
    }

    return report;
}

}  // namespace sensemap

#endif  // SENSEMAP_OPTIM_RANSAC_LORANSAC_H_
