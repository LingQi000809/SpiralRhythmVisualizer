import wave
import json

def get_wav_duration(filepath: str) -> float:
    with wave.open(filepath, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def adjust_start_times(json_path: str, output_path: str, audio1: str, audio2: str):
    # 1. compute durations
    dur1 = get_wav_duration(audio1)
    dur2 = get_wav_duration(audio2)
    total_duration = dur1 + dur2

    print(f"Duration 1: {dur1:.3f}s")
    print(f"Duration 2: {dur2:.3f}s")
    print(f"Total offset: {total_duration:.3f}s")

    # 2. load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 3. adjust startTime
    for item in data:
        if "startTime" in item:
            item["startTime"] += total_duration

    # 4. save updated JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return total_duration

# adjust_start_times(
#     json_path="./data/midi_pattern.json",
#     output_path="./data/midi_pattern_adjusted.json",
#     audio1="./data/midi_melody.wav",
#     audio2="./data/janeDoe_opening.wav"
# )



print(f"Duration: {get_wav_duration("./mono_poly.wav"):.3f}s")