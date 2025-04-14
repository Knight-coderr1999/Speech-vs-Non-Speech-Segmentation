# instantiate the pipeline
from pyannote.audio import Pipeline
diarization_pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="")



def diarize_audio(input_path, output_path):
    diarization = diarization_pipeline(input_path)
    with open(output_path, "w") as f:
        diarization.write_rttm(f)
