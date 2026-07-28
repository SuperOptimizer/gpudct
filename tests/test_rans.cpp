#include "core/models.hpp"
#include "core/rans.hpp"
#include "test.hpp"
#include "volumes.hpp"

using namespace gpudct::detail;
using gpudct_test::Rng;

namespace {

// Round-trips a symbol sequence through the interleaved coder and checks it
// comes back identical.
void round_trip(const std::vector<Sym>& syms, const std::vector<Model>& models,
                std::uint32_t nstates) {
  RansEncoder enc(nstates);
  const std::vector<std::uint8_t> bytes = enc.encode(syms, models);

  RansDecoder dec;
  REQUIRE(dec.init(bytes, nstates));
  for (std::size_t i = 0; i < syms.size(); ++i) {
    std::uint32_t v = 0;
    REQUIRE(dec.decode(models[syms[i].model], v));
    if (v != syms[i].value) {
      CHECK_EQ(v, syms[i].value);
      return;  // one desync makes every later symbol garbage; do not spam
    }
  }
}

}  // namespace

TEST(model_normalizes_to_the_probability_scale) {
  Rng rng(3);
  for (int trial = 0; trial < 200; ++trial) {
    const std::size_t n = 2 + (rng.next() % 255);
    std::vector<std::uint32_t> counts(n);
    for (auto& c : counts) c = (rng.next() % 1000);
    const Model m = Model::from_counts(counts);
    CHECK_EQ(m.nsym(), static_cast<std::uint32_t>(n));
    std::uint32_t sum = 0;
    for (std::uint32_t i = 0; i < m.nsym(); ++i) {
      // Every symbol keeps at least one slot, so a symbol the corpus never saw
      // is still encodable rather than a runtime failure on real data.
      CHECK(m.freq(i) >= 1);
      sum += m.freq(i);
    }
    CHECK_EQ(sum, kProbScale);
    CHECK_EQ(m.cum(0), 0);
  }
}

TEST(model_slot_table_inverts_the_cumulative_table) {
  std::vector<std::uint32_t> counts = {5, 100, 1, 40, 900, 3};
  const Model m = Model::from_counts(counts);
  for (std::uint32_t s = 0; s < kProbScale; ++s) {
    const std::uint32_t sym = m.sym_of_slot(s);
    CHECK(s >= m.cum(sym));
    CHECK(s < static_cast<std::uint32_t>(m.cum(sym)) + m.freq(sym));
  }
}

TEST(model_handles_all_zero_counts) {
  std::vector<std::uint32_t> counts(7, 0);
  const Model m = Model::from_counts(counts);
  std::uint32_t sum = 0;
  for (std::uint32_t i = 0; i < m.nsym(); ++i) sum += m.freq(i);
  CHECK_EQ(sum, kProbScale);
}

TEST(rans_round_trips_uniform_symbols) {
  Rng rng(11);
  std::vector<std::uint32_t> counts(16, 1);
  std::vector<Model> models = {bypass_model(), Model::from_counts(counts)};
  std::vector<Sym> syms;
  for (int i = 0; i < 5000; ++i) syms.push_back({1, static_cast<std::uint16_t>(rng.next() % 16)});
  round_trip(syms, models, 32);
}

// A near-deterministic source is where a coder with an off-by-one in its
// renormalization window falls over.
TEST(rans_round_trips_a_skewed_source) {
  Rng rng(12);
  std::vector<std::uint32_t> counts(4, 1);
  counts[0] = 1000000;
  std::vector<Model> models = {bypass_model(), Model::from_counts(counts)};
  std::vector<Sym> syms;
  for (int i = 0; i < 20000; ++i) {
    const std::uint16_t v = (rng.next() % 1000 == 0)
                                ? static_cast<std::uint16_t>(1 + rng.next() % 3)
                                : 0;
    syms.push_back({1, v});
  }
  round_trip(syms, models, 32);
}

TEST(rans_round_trips_mixed_models_and_bypass) {
  Rng rng(13);
  std::vector<Model> models = {bypass_model()};
  for (int k = 0; k < 6; ++k) {
    std::vector<std::uint32_t> counts(2 + static_cast<std::size_t>(k) * 40);
    for (std::size_t i = 0; i < counts.size(); ++i)
      counts[i] = 1 + (rng.next() % (1 + static_cast<std::uint32_t>(i)));
    models.push_back(Model::from_counts(counts));
  }
  std::vector<Sym> syms;
  for (int i = 0; i < 30000; ++i) {
    if (rng.next() % 3 == 0) {
      syms.push_back({kBypassModel, static_cast<std::uint16_t>(rng.next() & 1)});
    } else {
      const std::uint16_t mi = static_cast<std::uint16_t>(1 + rng.next() % 6);
      syms.push_back({mi, static_cast<std::uint16_t>(rng.next() % models[mi].nsym())});
    }
  }
  round_trip(syms, models, 32);
}

TEST(rans_round_trips_at_every_interleave_width) {
  Rng rng(14);
  std::vector<std::uint32_t> counts(20);
  for (std::size_t i = 0; i < counts.size(); ++i) counts[i] = static_cast<std::uint32_t>(100 - i * 4);
  std::vector<Model> models = {bypass_model(), Model::from_counts(counts)};
  std::vector<Sym> syms;
  for (int i = 0; i < 4000; ++i) syms.push_back({1, static_cast<std::uint16_t>(rng.next() % 20)});
  for (std::uint32_t n : {1u, 2u, 3u, 4u, 8u, 16u, 32u, 64u}) round_trip(syms, models, n);
}

TEST(rans_round_trips_an_empty_stream) {
  std::vector<Model> models = {bypass_model()};
  round_trip({}, models, 32);
}

// Fewer symbols than states: every state still has to be flushed and recovered.
TEST(rans_round_trips_fewer_symbols_than_states) {
  std::vector<std::uint32_t> counts(8, 1);
  std::vector<Model> models = {bypass_model(), Model::from_counts(counts)};
  std::vector<Sym> syms;
  for (std::uint16_t i = 0; i < 5; ++i) syms.push_back({1, i});
  round_trip(syms, models, 32);
}

TEST(model_serialization_round_trips) {
  Rng rng(15);
  std::vector<std::uint32_t> counts(37);
  for (auto& c : counts) c = rng.next() % 5000;
  const Model m = Model::from_counts(counts);

  std::vector<std::uint8_t> blob;
  m.serialize(blob);
  std::size_t pos = 0;
  Model back;
  REQUIRE(Model::deserialize(blob, pos, back));
  CHECK_EQ(pos, blob.size());
  CHECK_EQ(back.nsym(), m.nsym());
  for (std::uint32_t i = 0; i < m.nsym(); ++i) CHECK_EQ(back.freq(i), m.freq(i));
}

TEST(model_deserialization_rejects_garbage) {
  // Frequencies that do not sum to the probability scale would desync the
  // decoder silently; they must be rejected at parse time instead.
  std::vector<std::uint8_t> blob = {2, 0, 1, 0, 1, 0};
  std::size_t pos = 0;
  Model m;
  CHECK(!Model::deserialize(blob, pos, m));

  std::vector<std::uint8_t> truncated = {5, 0, 1};
  pos = 0;
  CHECK(!Model::deserialize(truncated, pos, m));
}

TEST(model_set_serialization_round_trips) {
  const ModelSet s = ModelSet::defaults();
  std::vector<std::uint8_t> blob;
  s.serialize(blob);
  std::size_t pos = 0;
  ModelSet back;
  REQUIRE(ModelSet::deserialize(blob, pos, back));
  CHECK_EQ(pos, blob.size());
  for (std::uint16_t i = 0; i < ModelIndex::total; ++i) {
    CHECK_EQ(back[i].nsym(), s[i].nsym());
    for (std::uint32_t v = 0; v < s[i].nsym(); ++v) CHECK_EQ(back[i].freq(v), s[i].freq(v));
  }
}

TEST(decoder_rejects_a_stream_too_short_for_its_states) {
  // Each interleaved state is flushed as 4 bytes, so a stream shorter than
  // 4 * nstates cannot even be initialized.
  RansDecoder dec;
  std::vector<std::uint8_t> tiny(8, 0);
  CHECK(!dec.init(tiny, 32));
  std::vector<std::uint8_t> just_short(127, 0);
  CHECK(!dec.init(just_short, 32));
  std::vector<std::uint8_t> just_enough(128, 0);
  CHECK(dec.init(just_enough, 32));
  // Byte-renormalized streams have no length-parity requirement.
  std::vector<std::uint8_t> odd(9, 0);
  CHECK(dec.init(odd, 1));
}

TEST_MAIN()
