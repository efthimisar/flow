#include "RecorderEngine.h"
#include <cmath>

// ---------------------------------------------------------------- WAV 24-bit με κεφαλίδα που ενημερώνεται τακτικά
class RecorderEngine::WavWriter
{
public:
    WavWriter (const juce::File& f, double sr) : stream (f), sampleRate ((uint32_t) std::lround (sr))
    {
        if (stream.failedToOpen()) return;
        stream.setPosition (0);
        stream.truncate();
        writeHeader (0);
    }

    bool ok() const { return ! stream.failedToOpen() && stream.getStatus().wasOk(); }

    bool write (const float* l, const float* r, int n)
    {
        bytes.ensureSize ((size_t) n * 6, false);
        auto* p = static_cast<uint8_t*> (bytes.getData());
        auto conv = [&p] (float v)
        {
            const float c = juce::jlimit (-1.0f, 1.0f, v);
            const int32_t s = (int32_t) std::lrintf (c * 8388607.0f);
            *p++ = (uint8_t) (s & 0xff); *p++ = (uint8_t) ((s >> 8) & 0xff); *p++ = (uint8_t) ((s >> 16) & 0xff);
        };
        for (int i = 0; i < n; ++i) { conv (l[i]); conv (r[i]); }
        if (! stream.write (bytes.getData(), (size_t) n * 6)) return false;
        dataBytes += (uint64_t) n * 6;
        return true;
    }

    // γράφει τα πραγματικά μεγέθη στην κεφαλίδα (ασφαλές ακόμη κι αν το host κρασάρει μετά)
    bool sync()
    {
        const auto pos = stream.getPosition();
        const bool a = writeHeader ((uint32_t) juce::jmin<uint64_t> (dataBytes, 0xffffffffu - 36));
        const bool b = stream.setPosition (pos);
        stream.flush();
        return a && b && stream.getStatus().wasOk();
    }

private:
    bool writeHeader (uint32_t dataSize)
    {
        if (! stream.setPosition (0)) return false;
        stream.write ("RIFF", 4);  stream.writeInt (36 + (int) dataSize);
        stream.write ("WAVE", 4);  stream.write ("fmt ", 4);
        stream.writeInt (16);      stream.writeShort (1);          // PCM
        stream.writeShort (2);     stream.writeInt ((int) sampleRate);
        stream.writeInt ((int) (sampleRate * 6)); stream.writeShort (6); stream.writeShort (24);
        stream.write ("data", 4);  stream.writeInt ((int) dataSize);
        return stream.getStatus().wasOk();
    }

    juce::FileOutputStream stream;
    uint32_t sampleRate;
    uint64_t dataBytes = 0;
    juce::MemoryBlock bytes;
};

// ---------------------------------------------------------------- RecorderEngine
RecorderEngine::RecorderEngine() : juce::Thread ("FlowRecorderWriter") {}

RecorderEngine::~RecorderEngine()
{
    stopWorker();
    if (writer != nullptr) finishTake (false);
}

void RecorderEngine::startWorker() { startThread (juce::Thread::Priority::normal); }
void RecorderEngine::stopWorker()  { stopThread (30000); }

void RecorderEngine::prepare (double sr)
{
    if (prepared.load() && std::abs (sr - sampleRate.load()) < 0.5) return;   // ίδιο sample rate: συνεχίζουμε

    const bool wasPrepared = prepared.exchange (false);
    stopWorker();
    if (wasPrepared && writer != nullptr) { finishTake (false); resetArmed(); }   // άλλαξε το sample rate μέσα σε λήψη

    sampleRate = sr;
    const int cap = (int) std::ceil (4.0 * sr);
    fifo.setTotalSize (cap);
    fifo.reset();
    fifoBuf.setSize (kChannels, cap);
    fifoBuf.clear();
    midiRing.assign (16384, MidiEv());
    midiFifo.setTotalSize ((int) midiRing.size());
    midiFifo.reset();
    infoRing.assign (4096, InfoEv());
    infoFifo.setTotalSize ((int) infoRing.size());
    infoFifo.reset();
    lastBpm = -1.0; lastScene = -1;
    for (int b = 0; b < kStems; ++b) { lastOff[b] = lastStop[b] = lastMute[b] = -1; }
    framesPushed = 0; framesConsumed = 0;
    preRollLen = (int) std::ceil (kPreRollSeconds * sr);
    preRoll.setSize (kChannels, preRollLen);
    preRoll.clear();
    preRollPos = preRollFilled = 0;
    chunk.setSize (kChannels, 4096);
    maxFrames = (int64_t) std::llround (kMaxTakeSeconds * sr);
    thresholdLin = (float) std::pow (10.0, kThresholdDb / 20.0);
    if (getState() == State::Armed) { framesWritten = 0; }

    prepared = true;
    startWorker();
}

void RecorderEngine::release() {}   // το αρχείο κλείνει μόνο με stop, TAPE FULL ή καταστροφή

static void raisePeak (std::atomic<float>& a, float v) { if (v > a.load (std::memory_order_relaxed)) a.store (v, std::memory_order_relaxed); }

void RecorderEngine::process (const juce::AudioBuffer<float>& buffer, const juce::AudioBuffer<float>& stems, uint32_t stemMask)
{
    if (! prepared.load (std::memory_order_acquire)) return;
    const int n = buffer.getNumSamples(), nCh = buffer.getNumChannels();
    if (n <= 0 || nCh <= 0) return;

    const int chR = nCh > 1 ? 1 : 0;
    raisePeak (peakL, buffer.getMagnitude (0, 0, n));
    raisePeak (peakR, buffer.getMagnitude (chR, 0, n));
    stemMaskNow.store (stemMask, std::memory_order_relaxed);

    int s1, n1, s2, n2;
    fifo.prepareToWrite (n, s1, n1, s2, n2);
    if (n1 + n2 < n) { dropped.fetch_add (n); return; }                     // γεμάτο FIFO (π.χ. κόλλησε ο δίσκος)

    auto put = [&] (int dstCh, const float* src)
    {
        if (n1 > 0) fifoBuf.copyFrom (dstCh, s1, src, n1);
        if (n2 > 0) fifoBuf.copyFrom (dstCh, s2, src + n1, n2);
    };
    put (0, buffer.getReadPointer (0));
    put (1, buffer.getReadPointer (chR));
    const int sc = juce::jmin (stems.getNumChannels(), 2 * kStems);
    for (int c = 0; c < 2 * kStems; ++c)
    {
        if (c < sc && stems.getNumSamples() >= n) put (2 + c, stems.getReadPointer (c));
        else
        {
            if (n1 > 0) fifoBuf.clear (2 + c, s1, n1);
            if (n2 > 0) fifoBuf.clear (2 + c, s2, n2);
        }
    }
    fifo.finishedWrite (n1 + n2);
    framesPushed.fetch_add (n, std::memory_order_relaxed);
}

void RecorderEngine::recordMidi (int block, const juce::MidiBuffer& events)
{
    if (! prepared.load (std::memory_order_acquire) || getState() != State::Recording || block < 0 || block >= kStems) return;
    const int64_t base = framesPushed.load (std::memory_order_relaxed);   // αρχή του τρέχοντος block ήχου (το process() του ίδιου block έρχεται μετά)
    for (const auto meta : events)
    {
        const auto m = meta.getMessage();
        const int n = m.getRawDataSize();
        if (n < 2 || n > 3) continue;
        const auto* d = m.getRawData();
        const int st = d[0] & 0xf0;
        const bool keep = st == 0x80 || st == 0x90 || st == 0xe0 || st == 0xd0 || st == 0xa0 || (st == 0xb0 && d[1] < 120);   // νότες, pitch bend, πίεση, CC (όχι all notes/sound off)
        if (! keep) continue;
        int s1, n1, s2, n2;
        midiFifo.prepareToWrite (1, s1, n1, s2, n2);
        if (n1 + n2 < 1) { midiDropped.fetch_add (1); continue; }
        auto& e = midiRing[(size_t) (n1 > 0 ? s1 : s2)];
        e.frame = base + meta.samplePosition; e.block = (uint8_t) block; e.b0 = d[0]; e.b1 = d[1]; e.b2 = n > 2 ? d[2] : 0;
        midiFifo.finishedWrite (1);
    }
}

void RecorderEngine::trackState (double bpm, const int* off, const int* stopped, const int* mute, int scene)
{
    if (! prepared.load (std::memory_order_acquire)) return;
    const int64_t base = framesPushed.load (std::memory_order_relaxed);
    auto push = [&] (uint8_t type, int block, int iv, float fv)
    {
        int s1, n1, s2, n2;
        infoFifo.prepareToWrite (1, s1, n1, s2, n2);
        if (n1 + n2 < 1) return;
        auto& e = infoRing[(size_t) (n1 > 0 ? s1 : s2)];
        e.frame = base; e.type = type; e.block = (uint8_t) block; e.iv = iv; e.fv = fv;
        infoFifo.finishedWrite (1);
    };
    if (std::abs (bpm - lastBpm) > 0.005) { lastBpm = bpm; push (0, 0, 0, (float) bpm); }
    for (int b = 0; b < kStems; ++b)
    {
        if (off[b]     != lastOff[b])  { lastOff[b]  = off[b];     push (1, b, off[b], 0.0f); }
        if (mute[b]    != lastMute[b]) { lastMute[b] = mute[b];    push (2, b, mute[b], 0.0f); }
        if (stopped[b] != lastStop[b]) { lastStop[b] = stopped[b]; push (3, b, stopped[b], 0.0f); }
    }
    if (scene != lastScene) { lastScene = scene; push (4, 0, scene, 0.0f); }
}

void RecorderEngine::drainInfo (bool atStart)
{
    const bool rec = atStart || (getState() == State::Recording && writer != nullptr);
    for (;;)
    {
        const int ready = infoFifo.getNumReady();
        if (ready <= 0) break;
        int s1, n1, s2, n2;
        infoFifo.prepareToRead (ready, s1, n1, s2, n2);
        auto take = [&] (int start, int cnt)
        {
            for (int i = 0; i < cnt; ++i)
            {
                const auto& e = infoRing[(size_t) (start + i)];
                if (rec && e.frame >= takeStartFrame) { takeInfo.push_back (e); continue; }
                if (e.block >= kStems) continue;
                switch (e.type)   // ό,τι έγινε πριν από το take ορίζει την αρχική κατάσταση
                {
                    case 0: curBpm = e.fv; break;
                    case 1: curOff[e.block] = e.iv; break;
                    case 2: curMute[e.block] = e.iv; break;
                    case 3: curStop[e.block] = e.iv; break;
                    case 4: curScene = e.iv; break;
                    default: break;
                }
            }
        };
        take (s1, n1); take (s2, n2);
        infoFifo.finishedRead (n1 + n2);
    }
}

static juce::var newObj() { return juce::var (new juce::DynamicObject()); }
static void setP (juce::var& o, const char* k, const juce::var& v) { o.getDynamicObject()->setProperty (k, v); }
static juce::String blockFile (int b, const char* ext) { return "Block " + juce::String (b + 1).paddedLeft ('0', 2) + ext; }

juce::var RecorderEngine::buildPaintJson()
{
    const double sr = juce::jmax (1.0, sampleRate.load());
    const juce::var info = infoEnd, infoS = infoStart;
    static const char* groupNames[] = { "percussion", "bass_arp", "pads_strings", "keys", "solo", "fx" };
    static const char* seqNames[] = { "auto", "flow", "plugin" };
    (void) seqNames;

    // tempo map: θέση σε samples και beats
    struct Seg { int64_t f; double bpm, beat; };
    std::vector<Seg> segs { { 0, initBpm, 0.0 } };
    juce::Array<juce::var> changes;
    for (const auto& e : takeInfo)
    {
        if (e.type != 0) continue;
        const int64_t rel = e.frame - takeStartFrame;
        const auto& pv = segs.back();
        const double beat = pv.beat + (double) (rel - pv.f) / sr * pv.bpm / 60.0;
        segs.push_back ({ rel, (double) e.fv, beat });
        auto c = newObj(); setP (c, "sample", (juce::int64) rel); setP (c, "beat", beat); setP (c, "bpm", (double) e.fv);
        changes.add (c);
    }
    auto beatAt = [&] (int64_t rel)
    {
        const Seg* sg = &segs[0];
        for (const auto& x : segs) if (x.f <= rel) sg = &x;
        return sg->beat + (double) (rel - sg->f) / sr * sg->bpm / 60.0;
    };

    auto hasPlugin = [&] (int b) { return (bool) info["blocks"][b]["plugin"]; };

    // γεγονότα ανά block + σκηνές (με τα ενεργά blocks εκείνη τη στιγμή)
    juce::Array<juce::var> mute[kStems];
    int sOff[kStems], sStop[kStems], sMute[kStems], sScene = initScene;
    for (int b = 0; b < kStems; ++b)
    {
        sOff[b] = initOff[b]; sStop[b] = initStop[b]; sMute[b] = initMute[b];
        auto add = [&] (const char* kind, bool muted, int mask, int64_t rel, const char* src)
        {
            auto m = newObj(); setP (m, "sample", (juce::int64) rel); setP (m, "beat", beatAt (rel)); setP (m, "kind", kind);
            setP (m, "muted", muted); if (mask >= 0) setP (m, "mask", mask); setP (m, "source", src);
            mute[b].add (m);
        };
        if (initOff[b])  add ("block_off", true, -1, 0, "initial");
        if (initStop[b]) add ("stopped", true, -1, 0, "initial");
        if (initMute[b]) add ("layer_mute", true, initMute[b], 0, "initial");
    }
    auto activeList = [&]
    {
        juce::Array<juce::var> a;
        for (int b = 0; b < kStems; ++b) if (hasPlugin (b) && ! sOff[b] && ! sStop[b]) a.add (b + 1);
        return a;
    };
    juce::Array<juce::var> arrangement;
    {
        auto a = newObj(); setP (a, "beat", 0.0); setP (a, "sample", (juce::int64) 0); setP (a, "event", "scene");
        setP (a, "scene", initScene + 1); setP (a, "name", juce::var()); setP (a, "active_blocks", activeList()); setP (a, "source", "initial");
        arrangement.add (a);
    }
    for (const auto& e : takeInfo)
    {
        const int64_t rel = e.frame - takeStartFrame;
        if (e.type == 0 || e.block >= kStems) continue;
        const int b = e.block;
        auto add = [&] (const char* kind, bool muted, int mask)
        {
            auto m = newObj(); setP (m, "sample", (juce::int64) rel); setP (m, "beat", beatAt (rel)); setP (m, "kind", kind);
            setP (m, "muted", muted); if (mask >= 0) setP (m, "mask", mask); setP (m, "source", "flow");
            mute[b].add (m);
        };
        if (e.type == 1) { sOff[b] = e.iv; add ("block_off", e.iv != 0, -1); }
        else if (e.type == 2) { sMute[b] = e.iv; add ("layer_mute", e.iv != 0, e.iv); }
        else if (e.type == 3) { sStop[b] = e.iv; add ("stopped", e.iv != 0, -1); }
        else if (e.type == 4)
        {
            sScene = e.iv;
            auto a = newObj(); setP (a, "beat", beatAt (rel)); setP (a, "sample", (juce::int64) rel); setP (a, "event", "scene");
            setP (a, "scene", e.iv + 1); setP (a, "name", juce::var()); setP (a, "active_blocks", activeList()); setP (a, "source", "flow");
            arrangement.add (a);
        }
    }
    (void) sScene; (void) sMute;

    juce::Array<juce::var> blocks;
    for (int b = 0; b < kStems; ++b)
    {
        const bool inTake = (takeMask >> b) & 1u;
        if (! hasPlugin (b) && ! inTake) continue;
        int notes = 0;
        for (const auto& e : takeMidi[b]) if ((e.b0 & 0xf0) == 0x90 && e.b2 > 0) ++notes;

        auto B = newObj();
        setP (B, "index", b + 1);
        setP (B, "name", "Block " + juce::String (b + 1).paddedLeft ('0', 2));
        const int g = (int) info["blocks"][b]["group"];
        setP (B, "group", info["blocks"][b]["group"].isVoid() || g < 0 || g > 5 ? juce::var() : juce::var (juce::String (groupNames[g])));
        setP (B, "role", juce::var());
        auto ins = newObj(); setP (ins, "type", juce::var()); setP (ins, "name", info["blocks"][b]["name"]); setP (ins, "preset", juce::var());
        setP (B, "instrument", ins);
        setP (B, "stem", inTake && stemFile[b].existsAsFile() ? juce::var (blockFile (b, ".wav")) : juce::var());
        setP (B, "stem_dry", juce::var());
        setP (B, "midi", notes > 0 ? juce::var (blockFile (b, ".mid")) : juce::var());
        setP (B, "midi_note_on_count", notes);
        setP (B, "midi_note", juce::var());
        setP (B, "pan", juce::var());
        setP (B, "volume_db", infoS["blocks"][b]["volume_db"]);
        setP (B, "mute_events", mute[b]);
        setP (B, "loop", info["blocks"][b]["loop"]);
        setP (B, "fx", juce::var());
        setP (B, "fx_changes", juce::var());
        setP (B, "fx_chain", info["blocks"][b]["fx_chain"]);

        const int sq = (int) info["blocks"][b]["seq"];
        juce::var seq; bool inferred = false;
        if (sq == 1) seq = "flow";
        else if (sq == 2) seq = "plugin";
        else { inferred = true; if (notes > 0) seq = "flow"; else if (takeStemPeak[b] > 0.003f) seq = "plugin"; }
        setP (B, "sequencer", seq);
        setP (B, "sequencer_inferred", inferred);
        blocks.add (B);
    }

    auto root = newObj();
    setP (root, "version", 1);
    setP (root, "flow_version", info["flow_version"]);
    auto rec = newObj();
    setP (rec, "kind", "rec_audio"); setP (rec, "sample_rate", sr); setP (rec, "bit_depth", 24);
    setP (rec, "length_samples", (juce::int64) takeLength); setP (rec, "start_wallclock", takeStartTime.toISO8601 (true));
    setP (rec, "latency_samples", juce::var());
    setP (root, "recording", rec);
    auto tempo = newObj();
    setP (tempo, "bpm", initBpm);
    juce::Array<juce::var> ts; ts.add (4); ts.add (4);
    setP (tempo, "time_signature", ts);
    setP (tempo, "changes", changes);
    setP (root, "tempo", tempo);
    auto master = newObj();
    setP (master, "file", "Master.wav"); setP (master, "reverb_send_return", juce::var()); setP (master, "limiter", info["limiter"]);
    setP (root, "master", master);
    setP (root, "blocks", blocks);
    setP (root, "arrangement", arrangement);
    return root;
}

void RecorderEngine::drainMidi()
{
    for (;;)
    {
        const int ready = midiFifo.getNumReady();
        if (ready <= 0) break;
        int s1, n1, s2, n2;
        midiFifo.prepareToRead (ready, s1, n1, s2, n2);
        const bool rec = getState() == State::Recording && writer != nullptr;
        auto take = [&] (int start, int cnt)
        {
            for (int i = 0; i < cnt; ++i)
            {
                const auto& e = midiRing[(size_t) (start + i)];
                if (rec && e.frame >= takeStartFrame && e.block < kStems) takeMidi[e.block].push_back (e);
            }
        };
        take (s1, n1); take (s2, n2);
        midiFifo.finishedRead (n1 + n2);
    }
}

bool RecorderEngine::writeMidiFile (int block, const juce::File& f)
{
    constexpr int ppq = 960;
    juce::MidiMessageSequence seq;
    const double sr = juce::jmax (1.0, sampleRate.load());
    const double ticksPerSec = takeBpm / 60.0 * ppq;
    seq.addEvent (juce::MidiMessage::tempoMetaEvent ((int) std::llround (60000000.0 / takeBpm)), 0.0);
    for (const auto& e : takeMidi[block])
    {
        const uint8_t raw[3] = { e.b0, e.b1, e.b2 };
        const int len = ((e.b0 & 0xf0) == 0xc0 || (e.b0 & 0xf0) == 0xd0) ? 2 : 3;
        seq.addEvent (juce::MidiMessage (raw, len, 0.0), std::round ((double) (e.frame - takeStartFrame) / sr * ticksPerSec));
    }
    seq.updateMatchedPairs();
    juce::MidiFile mf;
    mf.setTicksPerQuarterNote (ppq);
    mf.addTrack (seq);
    juce::TemporaryFile tmp (f);
    {
        juce::FileOutputStream out (tmp.getFile());
        if (! out.openedOk()) return false;
        out.setPosition (0); out.truncate();
        if (! mf.writeTo (out)) return false;
        out.flush();
    }
    return tmp.overwriteTargetFileWithTemporary();
}

void RecorderEngine::setFolder (const juce::File& f) { const juce::ScopedLock sl (fileLock); folder = f; }
juce::File RecorderEngine::getFolder() const          { const juce::ScopedLock sl (fileLock); return folder; }
juce::File RecorderEngine::getCurrentFile() const     { const juce::ScopedLock sl (fileLock); return currentFile; }
void RecorderEngine::pressButton()                    { pressRequested = true; notify(); }

juce::File RecorderEngine::makeTakeBase (const juce::File& dir, juce::Time t)
{
    const auto base = "Flow_" + t.formatted ("%Y-%m-%d_%H-%M");
    auto f = dir.getChildFile (base);
    for (int i = 2; (f.withFileExtension ("zip").exists() || f.withFileExtension ("wav").exists()) && i < 1000; ++i)
        f = dir.getChildFile (base + "_" + juce::String (i).paddedLeft ('0', 2));
    return f;
}

// ---------------------------------------------------------------- worker
void RecorderEngine::run()
{
    while (! threadShouldExit())
    {
        workerStep();
        wait (8);
    }
}

void RecorderEngine::resetArmed()
{
    preRollPos = preRollFilled = 0;
    framesWritten = 0;
    tapeFull = false;
    state = (int) State::Armed;
}

void RecorderEngine::enterError()
{
    writer.reset();
    for (auto& w : stemWriter) w.reset();
    state = (int) State::Error;
}

void RecorderEngine::closeWriters (bool& ok)
{
    if (writer != nullptr) { ok = writer->sync() && ok; writer.reset(); }
    for (auto& w : stemWriter)
        if (w != nullptr) { w->sync(); w.reset(); }
}

void RecorderEngine::packZip()
{
    zipping = true;
    juce::ZipFile::Builder builder;
    builder.addFile (masterFile, 0, "Master.wav");
    for (int b = 0; b < kStems; ++b)
    {
        int notes = 0;
        for (const auto& e : takeMidi[b]) if ((e.b0 & 0xf0) == 0x90 && e.b2 > 0) ++notes;
        if (notes > 0 && writeMidiFile (b, midiFile[b]))   // .mid μόνο όταν υπάρχει τουλάχιστον μία νότα
            builder.addFile (midiFile[b], 0, blockFile (b, ".mid"));
    }
    for (int b = 0; b < kStems; ++b)
        if ((takeMask >> b) & 1u && stemFile[b].existsAsFile())
            builder.addFile (stemFile[b], 0, blockFile (b, ".wav"));
    const auto paintFile = zipFile.getSiblingFile (zipFile.getFileNameWithoutExtension() + "_flow_paint.json");
    if (paintFile.replaceWithText (juce::JSON::toString (buildPaintJson(), false, 6)))
        builder.addFile (paintFile, 0, "flow_paint.json");

    bool ok = false;
    {
        juce::FileOutputStream out (zipFile);
        if (out.openedOk())
        {
            out.setPosition (0);
            out.truncate();
            ok = builder.writeToStream (out, nullptr);
            out.flush();
        }
    }

    if (ok)
    {
        // το Master.wav μένει και δίπλα στο zip
        for (int b = 0; b < kStems; ++b) { stemFile[b].deleteFile(); midiFile[b].deleteFile(); }
        paintFile.deleteFile();
    }
    {
        const juce::ScopedLock sl (fileLock);
        currentFile = ok ? zipFile : masterFile;
    }
    zipping = false;
}

void RecorderEngine::finishTake (bool full)
{
    drainMidi();   // τα τελευταία MIDI πριν κλείσει το take
    drainInfo (false);
    takeLength = framesWritten.load();
    { const juce::ScopedLock sl (infoLock); infoEnd = infoLatest; }
    bool ok = true;
    closeWriters (ok);
    if (ok) packZip();
    tapeFull = full;
    state = (int) (ok ? State::Stopped : State::Error);
}

void RecorderEngine::startTake()
{
    juce::File dir = getFolder();
    if (! dir.createDirectory().wasOk() || ! dir.hasWriteAccess()) { enterError(); return; }
    const auto base = makeTakeBase (dir, juce::Time::getCurrentTime());
    masterFile = base.withFileExtension ("wav");
    zipFile = base.withFileExtension ("zip");

    auto w = std::make_unique<WavWriter> (masterFile, sampleRate.load());
    if (! w->ok()) { enterError(); return; }

    for (auto& v : takeMidi) v.clear();
    takeInfo.clear();
    for (auto& pk : takeStemPeak) pk = 0.0f;
    drainMidi();                                  // ό,τι ήρθε πριν το πάτημα δεν ανήκει στο take
    takeStartFrame = framesConsumed;              // η πρώτη στιγμή του WAV
    drainInfo (true);
    initBpm = curBpm > 0.0 ? curBpm : 120.0;
    for (int b = 0; b < kStems; ++b) { initOff[b] = curOff[b]; initStop[b] = curStop[b]; initMute[b] = curMute[b]; }
    initScene = curScene;
    { const juce::ScopedLock sl (infoLock); infoStart = infoLatest; }
    takeStartTime = juce::Time::getCurrentTime();
    takeBpm = juce::jlimit (20.0, 400.0, tempoBpm.load());
    for (int b = 0; b < kStems; ++b) midiFile[b] = dir.getChildFile (base.getFileName() + "_block" + juce::String (b + 1).paddedLeft ('0', 2) + ".mid");
    takeMask = stemMaskNow.load();
    for (int b = 0; b < kStems; ++b)
    {
        if (! ((takeMask >> b) & 1u)) continue;
        stemFile[b] = dir.getChildFile (base.getFileName() + "_block" + juce::String (b + 1).paddedLeft ('0', 2) + ".wav");
        auto sw = std::make_unique<WavWriter> (stemFile[b], sampleRate.load());
        if (sw->ok()) stemWriter[b] = std::move (sw);
        else takeMask &= ~(1u << b);
    }

    { const juce::ScopedLock sl (fileLock); currentFile = masterFile; }
    writer = std::move (w);
    framesWritten = 0;
    framesSinceSync = 0;
    tapeFull = false;
    state = (int) State::Recording;
}

bool RecorderEngine::writeRange (const juce::AudioBuffer<float>& src, int start, int n)
{
    if (! writer->write (src.getReadPointer (0) + start, src.getReadPointer (1) + start, n)) return false;
    for (int b = 0; b < kStems; ++b)
        if (stemWriter[b] != nullptr)
            if (! stemWriter[b]->write (src.getReadPointer (2 + 2 * b) + start, src.getReadPointer (3 + 2 * b) + start, n))
                stemWriter[b].reset();   // ένα stem που απέτυχε δεν σταματά τη λήψη
    return true;
}

void RecorderEngine::workerStep()
{
    if (pressRequested.exchange (false))
    {
        switch (getState())
        {
            case State::Recording: finishTake (false); break;
            case State::Stopped:
            case State::Error:     resetArmed(); startTake(); break;   // νέο take: ξεκινά αμέσως στο πάτημα
            case State::Armed:     startTake();        break;          // ξεκινά αμέσως στο πάτημα, όχι με τον πρώτο ήχο
        }
    }

    drainMidi();
    drainInfo (false);
    for (;;)
    {
        const int ready = fifo.getNumReady();
        if (ready <= 0) break;
        const int n = juce::jmin (ready, chunk.getNumSamples());
        int s1, n1, s2, n2;
        fifo.prepareToRead (n, s1, n1, s2, n2);
        for (int c = 0; c < kChannels; ++c)
        {
            if (n1 > 0) chunk.copyFrom (c, 0, fifoBuf, c, s1, n1);
            if (n2 > 0) chunk.copyFrom (c, n1, fifoBuf, c, s2, n2);
        }
        fifo.finishedRead (n1 + n2);
        const int cnt = n1 + n2;
        framesConsumed += cnt;
        const float* l = chunk.getReadPointer (0);
        const float* r = chunk.getReadPointer (1);

        switch (getState())
        {
            case State::Armed:
                break;   // δεν ξεκινά μόνο του: περιμένει το πάτημα
            case State::Recording:
            {
                if (writer == nullptr) break;
                const int64_t remaining = maxFrames - framesWritten.load();
                const int todo = (int) juce::jmin<int64_t> (cnt, remaining);
                if (todo > 0)
                {
                    if (! writeRange (chunk, 0, todo)) { enterError(); break; }
                    for (int b = 0; b < kStems; ++b)
                        if (stemWriter[b] != nullptr)
                            takeStemPeak[b] = juce::jmax (takeStemPeak[b], chunk.getMagnitude (2 + 2 * b, 0, todo), chunk.getMagnitude (3 + 2 * b, 0, todo));
                    framesWritten += todo;
                    framesSinceSync += todo;
                }
                if (framesWritten.load() >= maxFrames) { finishTake (true); break; }     // TAPE FULL στα 12:00
                if (framesSinceSync >= (int64_t) (kSyncSeconds * sampleRate.load()))
                {
                    framesSinceSync = 0;
                    bool ok = writer->sync();
                    for (auto& w : stemWriter) if (w != nullptr) w->sync();
                    if (! ok) enterError();
                }
                break;
            }
            case State::Stopped:
            case State::Error:
                break;   // ο ήχος περνά αλλά δεν γράφεται
        }
    }
}
