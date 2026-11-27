#pragma once
#include "tasks/optimization_task.h"

namespace STreeD {

	// A custom solution class for MCC
	// counting false negatives and false positives; class totals recover TP and TN
	struct MCCSol {
		int false_negatives{ 0 };
		int false_positives{ 0 };

		inline bool operator==(const MCCSol& other) const {
			return other.false_negatives == false_negatives &&
				other.false_positives == false_positives;
		}

		inline bool operator!=(const MCCSol& other) const { return !(*this == other); }

		inline const MCCSol& operator+=(const MCCSol& v2) { false_negatives += v2.false_negatives; false_positives += v2.false_positives; return *this; }
		inline const MCCSol& operator-=(const MCCSol& v2) { false_negatives -= v2.false_negatives; false_positives -= v2.false_positives; return *this; }
		inline MCCSol operator+(const MCCSol& v2) const { return MCCSol(*this) += v2; }
		inline MCCSol operator-(const MCCSol& v2) const { return MCCSol(*this) -= v2; }

		inline MCCSol operator*(const int multiplier) { return MCCSol({ false_negatives * multiplier, false_positives * multiplier }); }
	};

	class MCC : public Classification {
	public:
		using SolType = MCCSol;
		using SolD2Type = MCCSol;
		using TestSolType = MCCSol;

		static const bool total_order = false;
		static const bool custom_leaf = false;
		static const bool check_unique = false;
		static const bool terminal_zero_costs_true_label = false; // True iff the costs of assigning the true label in the terminal is zero
		static constexpr MCCSol worst = { INT32_MAX, INT32_MAX };
		static constexpr MCCSol best = { 0, 0 };

		MCC(const ParameterHandler& parameters) : Classification(parameters) {}

        void InformTrainData(const ADataView& data, const DataSummary& summary);
        void InformTestData(const ADataView& data, const DataSummary& summary);

		MCCSol GetLeafCosts(const ADataView& data, const BranchContext& context, int label) const;
        MCCSol GetTestLeafCosts(const ADataView& data, const BranchContext& context, int label) const;
		inline void GetInstanceLeafD2Costs(const AInstance* instance, int org_label, int label, MCCSol& costs, int multiplier) const {
			costs = label == 0
				? MCCSol({ org_label == 1 ? multiplier : 0, 0 })
				: MCCSol({ 0, org_label == 0 ? multiplier : 0 });
		}
		inline bool IsD2ZeroCost(const MCCSol& d2costs) const { return d2costs.false_negatives == 0 && d2costs.false_positives == 0; }
		void ComputeD2Costs(const MCCSol& d2costs, int count, int label, int depth, MCCSol& costs) const { costs = d2costs; }
		inline MCCSol GetWorstPerLabel(int label) const { return label == 0 ? MCCSol({ 0, 1 }) : MCCSol({ 1, 0 }); }
		
		double ComputeTrainScore(const MCCSol& train_value) const;
		double ComputeTrainTestScore(const MCCSol& test_value) const;
		double ComputeTestTestScore(const MCCSol& test_value) const;

		inline static MCCSol Add(const MCCSol& left, const MCCSol& right) {
			return { left.false_negatives + right.false_negatives, left.false_positives + right.false_positives };
		}
		inline static MCCSol TestAdd(const MCCSol& left, const MCCSol& right) { return Add(left, right); }
		inline static void Add(const MCCSol& left, const MCCSol& right, MCCSol& out) {
			out.false_negatives = left.false_negatives + right.false_negatives;
			out.false_positives = left.false_positives + right.false_positives;
		}

		inline static void Subtract(const MCCSol& left, const MCCSol& right, MCCSol& out) {
			out = MCCSol({ std::max(0, left.false_negatives - right.false_negatives),
								std::max(0, left.false_positives - right.false_positives) });
		}

        // For fixed class totals, MCC is non-increasing in each error count.
        // Componentwise dominance is preserved when the same sibling is added.
		inline static bool Dominates(const MCCSol& s1, const MCCSol& s2) {
			return (s1.false_negatives <= s2.false_negatives && s1.false_positives <= s2.false_positives);

		}

		inline static bool DominatesInv(const MCCSol& s1, const MCCSol& s2) {
			return (s1.false_negatives >= s2.false_negatives && s1.false_positives >= s2.false_positives);
		}

		// (currently not used)
		inline static bool FrontLT(const MCCSol& s1, const MCCSol& s2) {
			return (s1.false_positives < s2.false_positives || (s1.false_positives == s2.false_positives && s1.false_negatives < s2.false_negatives));
		}

		inline bool DominatesD0(const MCCSol& s1, const MCCSol& s2) const {
			return (ComputeTrainScore(s1) >= ComputeTrainScore(s2));
		}

		inline bool DominatesD0Inv(const MCCSol& s1, const MCCSol& s2) const {
			return (ComputeTrainScore(s1) <= ComputeTrainScore(s2));
		}

		inline static double CalcDiff(const MCCSol& s1, const MCCSol& s2) {
            // Bound states can contain INT32_MAX: convert before subtracting/squaring.
            double fn = double(s1.false_negatives) - s2.false_negatives;
            double fp = double(s1.false_positives) - s2.false_positives;
            return fn * fn + fp * fp;
		}

		inline static void Merge(const MCCSol& s1, const MCCSol& s2, MCCSol& out) {
			out = { std::min(s1.false_negatives, s2.false_negatives), std::min(s1.false_positives, s2.false_positives) };
		}

		inline static void MergeInv(const MCCSol& s1, const MCCSol& s2, MCCSol& out) {
			out = { std::max(s1.false_negatives, s2.false_negatives), std::max(s1.false_positives, s2.false_positives) };
		}

		inline static std::vector<MCCSol> ExtremePoints() {
			return { {0, INT32_MAX}, {INT32_MAX, 0} };
		}
    private:
        static double ComputeScore(const MCCSol& value, const DataSummary& summary);
	};

}

namespace std {
	template <>
	struct hash<STreeD::MCCSol> {

		//adapted from https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
		size_t operator()(const STreeD::MCCSol& sol) const {
			using std::size_t;
			using std::hash;
			size_t seed = hash<int>()(sol.false_negatives);
			seed ^= hash<int>()(sol.false_positives) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			return seed;
		}
	};

}
