#include "../cbp.hpp"
#include "../harcom.hpp"

using namespace hcm;

template <u64 BHT_SIZE_BITS=6, u64 GHR_LEN=6>
struct global_predictor : predictor {
    static constexpr u64 BHT_SIZE = 1 << BHT_SIZE_BITS;
    static constexpr u64 MAX_SIZE = std::max(BHT_SIZE_BITS, GHR_LEN);
    ram<val<2>, BHT_SIZE> bht;
    reg<2> bh;
    reg<GHR_LEN> ghr;

    val<1> predict1([[maybe_unused]] val<64> inst_pc)
    {
        // Hash the instruction PC to a 6-bit index by first chunking it into
        // an array of 6-bit values, then folding that array onto itself using
        // XOR.
        val<MAX_SIZE> xored = inst_pc ^ ghr;
        val<BHT_SIZE_BITS> index = xored.make_array(val<BHT_SIZE_BITS>{}).fold_xor();
        bh = bht.read(index);

        // Use the top bit of the counter to predict the branch's direction
        return bh >> 1;
    };

    val<1> predict2([[maybe_unused]] val<64> inst_pc)
    {
        // re-use the same prediction for the second-level predictor
        return bh >> 1;
    }

    // Note: common.hpp contains a more generic version of a saturating counter
    // update, this is reproduced here for learning purposes.
    inline val<2> update_bh(val<2> counter, val<1> taken) {
        val<2> increased = select(counter == 3, counter, val<2>{counter + 1});
        val<2> decreased = select(counter == 0, counter, val<2>{counter - 1});
        return select(taken, increased, decreased);
    }


    inline void update_ghr(val<1> taken) {
        val<GHR_LEN> shifted = ghr << hard<1>{};
        ghr = shifted + taken;
    }

    void update_condbr([[maybe_unused]] val<64> branch_pc, [[maybe_unused]] val<1> taken, [[maybe_unused]] val<64> next_pc)
    {
        val<2> newbh = update_bh(bh, taken);
        val<1> performing_bh_update = val<1>{newbh != bh};

        need_extra_cycle(performing_bh_update);
        execute_if(performing_bh_update, [&](){
            val<MAX_SIZE> xored = branch_pc ^ ghr;
            val<BHT_SIZE_BITS> index = xored.make_array(val<BHT_SIZE_BITS>{}).fold_xor();
            bht.write(index, newbh);
        });

        update_ghr(taken);
    }

    void update_cycle([[maybe_unused]] instruction_info &block_end_info)
    {
    }

    // reuse_predict1 and reuse_predict2 will never be called because this
    // predictor never calls reuse_prediction()
    val<1> reuse_predict1([[maybe_unused]] val<64> inst_pc)
    {
        return hard<0>{};
    };
    val<1> reuse_predict2([[maybe_unused]] val<64> inst_pc)
    {
        return hard<0>{};
    }
};
