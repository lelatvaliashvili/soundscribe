#TODO: cache demucs
#TODO:either real time remix pipeline or CLAP semantic control
#TODO: “Right now the system uses a session-level state to detect remix feedback instead of reclassifying each prompt with the LLM. I’m wondering whether a hybrid approach would be better, where we check both session state and prompt intent.”
'''
arch suggestions:
  ↓
controller
   ↓
processing_pipeline
   ↓
audio_engine
   ↓
DSP modules

audio_engine/
   pipeline.py
   processors/
       gain_processor.py
       pitch_processor.py
       reverb_processor.py
       compression_processor.py

then remix becomes pipeline = AudioPipeline([
    GainProcessor(),
    PitchProcessor(),
    ReverbProcessor(),
    CompressorProcessor()
])

pipeline.process(stems, instructions)
audio_analysis/
    tempo
    loudness
    spectral centroid
    dynamic range
'''