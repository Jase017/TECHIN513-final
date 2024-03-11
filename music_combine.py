from pydub import AudioSegment

def mix_and_align_audio_with_fades(filename1_Instrument, filename2_Vocal, output_filename, adjust_db=3):
    # Construct the full file paths
    audio_path1_Instrument = f"D:\\Techin513\\Instrument\\{filename1_Instrument}"
    audio_path2_Vocal = f"D:\\Techin513\\Vocal\\{filename2_Vocal}"
    output_path = f"D:\\Techin513\\Output\\{output_filename}"
    
    # Load the two audio files
    audio1 = AudioSegment.from_file(audio_path1_Instrument)
    audio2 = AudioSegment.from_file(audio_path2_Vocal)

    # Adjust the volume: increase the vocals volume and decrease the instrument's volume
    audio1 = audio1 - adjust_db  # Decrease the instrument's volume
    audio2 = audio2 + adjust_db  # Increase the vocal's volume

    # Get the duration of both audio files
    duration1 = len(audio1)
    duration2 = len(audio2)

    # Calculate the duration difference and trim the longer audio to match the length of the shorter one
    if duration1 > duration2:
        audio1 = audio1[:duration2]  # If the instrument track is longer, trim it to match the vocal's length
    elif duration2 > duration1:
        audio2 = audio2[:duration1]  # If the vocal track is longer, trim it to match the instrument's length

    # Combine the two adjusted audio files
    mixed_audio = audio1.overlay(audio2)

    # Add fade-in and fade-out effects
    fade_in_duration = 5000  # 5 seconds fade-in
    fade_out_duration = 5000  # 5 seconds fade-out
    final_audio = mixed_audio.fade_in(fade_in_duration).fade_out(fade_out_duration)

    # Export the combined audio file
    final_audio.export(output_path, format="wav")

# Get the filenames from user input
print("Jazzhiphop(j1.wav--j10.wav)")
print("Boombap(b1.wav--b10.wav)")
print("Trap(t1.wav--t10.wav)")
print("Drill(d1.wav--d10.wav)")
filename1 = input("Please enter the name of the instrumental audio file (including extension): ")
filename2 = input("Please enter the name of the vocal audio file (including extension): ")
output_filename = "hiphopcombine.wav"

# Call the function
mix_and_align_audio_with_fades(filename1, filename2, output_filename)