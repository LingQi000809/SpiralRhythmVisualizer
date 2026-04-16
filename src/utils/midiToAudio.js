import * as Tone from "tone";
import { Midi } from "@tonejs/midi";

export async function renderMidiToAudio(midiData) {
    const midi = new Midi(midiData);

    // Create an offline context for the duration of the MIDI file
    return await Tone.Offline(async(context) => {
        const synth = new Tone.PolySynth(Tone.Synth).toDestination();

        midi.tracks.forEach((track) => {
            track.notes.forEach((note) => {
                synth.triggerAttackRelease(
                    note.name,
                    note.duration,
                    note.time,
                    note.velocity
                );
            });
        });
    }, midi.duration).then((buffer) => {
        // Convert the AudioBuffer to a Blob URL
        return audioBufferToBlobUrl(buffer);
    });
}

function audioBufferToBlobUrl(audioBuffer) {
    const wav = audioBufferToWav(audioBuffer);
    const blob = new Blob([wav], { type: "audio/wav" });
    return URL.createObjectURL(blob);
}

// Simple WAV encoder helper
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const bufferLen = buffer.length * blockAlign;
    const headerLen = 44;
    const arrayBuffer = new ArrayBuffer(headerLen + bufferLen);
    const view = new DataView(arrayBuffer);

    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + bufferLen, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, bufferLen, true);

    const offset = 44;
    const channelData = [];
    for (let i = 0; i < numChannels; i++) {
        channelData.push(buffer.getChannelData(i));
    }

    let index = 0;
    for (let i = 0; i < buffer.length; i++) {
        for (let channel = 0; channel < numChannels; channel++) {
            let sample = channelData[channel][i];
            sample = Math.max(-1, Math.min(1, sample));
            view.setInt16(offset + index, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
            index += bytesPerSample;
        }
    }
    return arrayBuffer;
}