#include "tasks/mcc.h"
#include <stdexcept>

using namespace STreeD;

static void Check(bool condition) {
    if (!condition) throw std::runtime_error("MCC task check failed");
}

static DataSummary Summary(int positives, int negatives) {
    DataSummary summary;
    summary.instances_per_class = { negatives, positives };
    summary.size = positives + negatives;
    summary.num_labels = 2;
    return summary;
}

int main() {
    auto parameters = ParameterHandler::DefineParameters();
    MCC task(parameters);
    ADataView unused;
    task.InformTrainData(unused, Summary(2, 2));
    Check(task.ComputeTrainScore({ 0, 0 }) == 1.0);
    Check(task.ComputeTrainScore({ 2, 2 }) == -1.0);
    Check(task.ComputeTrainScore({ 2, 0 }) == 0.0);
    Check(task.ComputeTrainScore({ 0, 2 }) == 0.0);
    Check(task.ComputeTrainScore({ 1, 1 }) == 0.0);
    Check(task.ComputeTrainTestScore({ 0, 0 }) == 1.0);
    Check(task.DominatesD0({ 0, 0 }, { 1, 1 }));
    Check(task.DominatesD0Inv({ 1, 1 }, { 0, 0 }));
    Check(task.ComputeTrainScore(MCC::worst) < -1.0);
    Check(std::isfinite(MCC::CalcDiff(MCC::worst, MCC::best)));

    task.InformTestData(unused, Summary(4, 6));
    Check(std::abs(task.ComputeTestTestScore({ 1, 2 }) - 10.0 / std::sqrt(600.0)) < 1e-12);
    task.InformTestData(unused, Summary(0, 6));
    Check(task.ComputeTestTestScore({ 0, 3 }) == 0.0);
    DataSummary only_zero;
    only_zero.instances_per_class = { 6 };
    task.InformTestData(unused, only_zero);
    Check(task.ComputeTestTestScore({ 0, 3 }) == 0.0);

    // Products exceed int32 range; the expected correlation remains exact.
    task.InformTrainData(unused, Summary(100000, 100000));
    Check(std::abs(task.ComputeTrainScore({ 25000, 25000 }) - 0.5) < 1e-12);

    // Verify the componentwise pruning order, including zero denominators.
    for (int p = 1; p <= 20; ++p) {
        for (int n = 1; n <= 20; ++n) {
            task.InformTrainData(unused, Summary(p, n));
            for (int fn = 0; fn <= p; ++fn) {
                for (int fp = 0; fp <= n; ++fp) {
                    const double score = task.ComputeTrainScore({ fn, fp });
                    if (fn < p) Check(score + 1e-12 >= task.ComputeTrainScore({ fn + 1, fp }));
                    if (fp < n) Check(score + 1e-12 >= task.ComputeTrainScore({ fn, fp + 1 }));
                }
            }
        }
    }
}
