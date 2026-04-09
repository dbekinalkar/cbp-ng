#include "../cbp.hpp"
#include "../harcom.hpp"

using namespace hcm;

/*
 * HIST_LEN: # of bits in each entry in the PHT
 * PHT_SIZE_BITS number of bits needed to index all ntries in the PHT
 * BHT_SIZE_BITS: number of bits needed to index all entries in the BHT (at most PHT_SIZE_BITS)
*/

template <u64 BHT_SIZE_BITS=6, u64 PHT_SIZE_BITS=12, u64 HIST_LEN=6>
struct local_predictor : predictor {

    static constexpr u64 BHT_SIZE = 1 << BHT_SIZE_BITS;
    static constexpr u64 PHT_SIZE = 1 << PHT_SIZE_BITS;
    
    ram<val<2>, BHT_SIZE> bht;
    ram<val<HIST_LEN>, PHT_SIZE> pht;

    reg<HIST_LEN> ph;
    reg<2> bh;

    val<1> predict1([[maybe_unused]] val<64> inst_pc)
    {
        // Hash the instruction PC to a 6-bit index by first chunking it into
        // an array of 6-bit values, then folding that array onto itself using
        // XOR.
        val<PHT_SIZE_BITS> ph_index = inst_pc.make_array(val<PHT_SIZE_BITS>{}).fold_xor();
        ph = pht.read(ph_index);
        // need_extra_cycle(hard<1>{});
        val<BHT_SIZE_BITS> bh_index = ph.make_array(val<BHT_SIZE_BITS>{}).fold_xor();
        bh = bht.read(bh_index);

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


    inline val<HIST_LEN> update_ph(val<HIST_LEN> ph, val<1> taken) {
        val<HIST_LEN> shifted = ph << hard<1>{};
        return shifted.fo1() + taken;
    }

    void update_condbr([[maybe_unused]] val<64> branch_pc, [[maybe_unused]] val<1> taken, [[maybe_unused]] val<64> next_pc)
    {
        // Update PHT
        val<HIST_LEN> newph = update_ph(ph, taken);
        val<1> performing_ph_update = val<1>{newph != ph};

        val<2> newbh = update_bh(bh, taken);
        val<1> performing_bh_update = val<1>{newbh != bh};

        val<1> performing_update = val<1>{performing_ph_update | performing_bh_update};

        need_extra_cycle(performing_update.fo1());
        execute_if(performing_ph_update, [&](){
            val<PHT_SIZE_BITS> ph_index = branch_pc.make_array(val<PHT_SIZE_BITS>{}).fold_xor();
            pht.write(ph_index, newph);
        });

        // Update BHT
        execute_if(performing_bh_update, [&](){
            val<BHT_SIZE_BITS> bh_index = ph.make_array(val<BHT_SIZE_BITS>{}).fold_xor();
            bht.write(bh_index, newbh);
        });
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
