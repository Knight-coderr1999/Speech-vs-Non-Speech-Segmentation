import matplotlib.pyplot as plt
from pyannote.core import Annotation, Segment

def plot_diarization_timeline(rttm_file):
    """
    Plot the speaker diarization timeline from an RTTM file.
    
    Parameters:
    rttm_file (str): Path to the RTTM file.
    """
    # Read the RTTM file
    annotation = Annotation()
    with open(rttm_file, "r") as f:
        for line in f:
            fields = line.strip().split()
            start_time = float(fields[3])
            duration = float(fields[4])
            speaker_label = fields[7]
            segment = Segment(start_time, start_time + duration)
            annotation[segment, 0] = speaker_label

    # Get unique speaker labels
    labels = sorted(set(label for _, _, label in annotation.itertracks(yield_label=True)))
    label_map = {label: i for i, label in enumerate(labels)}

    # Plot speaker timelines
    fig, ax = plt.subplots(figsize=(12, 2))
    for segment, _, label in annotation.itertracks(yield_label=True):
        ax.plot([segment.start, segment.end], [label_map[label]] * 2, linewidth=8)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time (s)")
    ax.set_title("Speaker Diarization Timeline")
    plt.tight_layout()
    plt.show()

# Example usage
annotation = Annotation()
with open("/content/audio_pipeline_dataset/denoised/id01106_id02725_mix_urbanmix_denoised.rttm", "r") as f:
    for line in f:
        fields = line.strip().split()
        start_time = float(fields[3])
        duration = float(fields[4])
        speaker_label = fields[7]
        segment = Segment(start_time, start_time + duration)
        annotation[segment, 0] = speaker_label

# Get unique speaker labels
labels = sorted(set(label for _, _, label in annotation.itertracks(yield_label=True)))
label_map = {label: i for i, label in enumerate(labels)}

# Plot speaker timelines
fig, ax = plt.subplots(figsize=(12, 2))
for segment, _, label in annotation.itertracks(yield_label=True):
    ax.plot([segment.start, segment.end], [label_map[label]] * 2, linewidth=8)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel("Time (s)")
ax.set_title("Speaker Diarization Timeline")
plt.tight_layout()
plt.show()
