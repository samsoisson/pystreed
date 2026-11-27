#include "tasks/mcc.h"
#include <stdexcept>

namespace STreeD {

    void MCC::InformTrainData(const ADataView& data, const DataSummary& summary) {
        if (summary.instances_per_class.size() != 2
            || summary.instances_per_class[0] == 0 || summary.instances_per_class[1] == 0) {
            throw std::invalid_argument("MCC training requires both binary labels 0 and 1.");
        }
        OptimizationTask::InformTrainData(data, summary);
    }

    void MCC::InformTestData(const ADataView& data, const DataSummary& summary) {
        if (summary.instances_per_class.size() > 2) {
            throw std::invalid_argument("MCC scoring requires binary labels 0 and 1.");
        }
        OptimizationTask::InformTestData(data, summary);
    }

    MCCSol MCC::GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const {
        return GetTestLeafCosts(data, context, label);
    }

    MCCSol MCC::GetTestLeafCosts(const ADataView& data, const BranchContext&, int label) const {
        // Evaluation data may contain only class 0, so index 1 need not exist.
        const int positives = data.NumLabels() > 1 ? data.NumInstancesForLabel(1) : 0;
        const int negatives = data.NumLabels() > 0 ? data.NumInstancesForLabel(0) : 0;
        return label == 0 ? MCCSol{ positives, 0 } : MCCSol{ 0, negatives };
    }

    double MCC::ComputeScore(const MCCSol& value, const DataSummary& summary) {
        const double positives = summary.instances_per_class.size() > 1
            ? summary.instances_per_class[1] : 0;
        const double negatives = summary.instances_per_class.empty()
            ? 0 : summary.instances_per_class[0];

        // Worst-solution and relaxed bound states are not confusion matrices.
        // They must rank below even a valid MCC of -1, without producing NaN.
        if (value.false_negatives < 0 || value.false_positives < 0
            || value.false_negatives > positives || value.false_positives > negatives) {
            return -std::numeric_limits<double>::infinity();
        }

        const double fn = value.false_negatives;
        const double fp = value.false_positives;
        const double tp = positives - fn;
        const double tn = negatives - fp;
        const double denominator = (tp + fp) * positives * negatives * (tn + fn);
        // Same convention as sklearn: constant predictions or true labels score 0.
        if (denominator == 0) return 0.0;
        const double score = (tp * tn - fp * fn) / std::sqrt(denominator);
        return std::clamp(score, -1.0, 1.0);
    }

    double MCC::ComputeTrainScore(const MCCSol& value) const {
        return ComputeScore(value, train_summary);
    }

    double MCC::ComputeTrainTestScore(const MCCSol& value) const {
        return ComputeTrainScore(value);
    }

    double MCC::ComputeTestTestScore(const MCCSol& value) const {
        return ComputeScore(value, test_summary);
    }

}
