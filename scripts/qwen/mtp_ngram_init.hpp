// Developer-only visibility into logical cache state; no layout changes.
#pragma once
#include "engine/model.hpp"
#include <stdexcept>
namespace zerocool::engine {
struct NgramAudit {
    static Json summary(const NgramStore& s) {
        return {{"capacity_rows",s.rows_.capacity()},{"constructed_rows",s.rows_.size()},
            {"cached_rows",s.lookup_.size()},{"next_slot",s.next_},
            {"reserved_row_bytes",s.rows_.capacity()*sizeof(NgramStore::Row)},
            {"initialized_row_bytes",s.rows_.size()*sizeof(NgramStore::Row)},
            {"hits",s.hits},{"misses",s.misses}};
    }
    static Json summary(const Model& m) {return summary(*m.ngrams_);}
    static bool same(const NgramStore& a,const NgramStore& b) {
        if(a.rows_.capacity()!=b.rows_.capacity() || a.lookup_!=b.lookup_ || a.next_!=b.next_ ||
            a.hits!=b.hits || a.misses!=b.misses) return false;
        for(const auto& [key,slot]:a.lookup_) {
            if(slot>=a.rows_.size() || slot>=b.rows_.size()) return false;
            if(a.rows_[slot].key!=key || b.rows_[slot].key!=key || a.rows_[slot].values!=b.rows_[slot].values) return false;
        }
        return true;
    }
};
}
