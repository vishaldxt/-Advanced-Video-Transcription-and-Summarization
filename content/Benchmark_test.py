from fastapi import FastAPI, HTTPException
from pytube import YouTube
import numpy as np
import wave
import subprocess
import os
import psutil
import time
import logging
from typing import List
from dotenv import load_dotenv
from starlette.responses import FileResponse
from deepspeech import Model
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
import numpy as np
import io
# Load environment variables
load_dotenv()

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Model paths and directories
model_file_path = '/app/Models/deepspeech-0.9.3-models.pbmm'
scorer_file_path = '/app/Models/deepspeech-0.9.3-models.scorer'
content_directory = '/app/content/'
downloads_path = '/app/downloads'

# Load DeepSpeech model
model = Model(model_file_path)
model.enableExternalScorer(scorer_file_path)


async def async_benchmark(task_function, *args, **kwargs):
    try:
        start_time = time.time()
        psutil.cpu_percent(interval=None)  # Initial call to set the value
        start_memory = psutil.virtual_memory().used

        result = await task_function(*args, **kwargs)  # Await the async task function

        end_cpu = psutil.cpu_percent(interval=1)  # Measure over a 1-second period
        end_time = time.time()
        end_memory = psutil.virtual_memory().used

        time_elapsed = end_time - start_time
        cpu_used = end_cpu  # This is already a percentage
        raw_memory_used = end_memory - start_memory
        memory_used = max(raw_memory_used, 0)  # Avoid negative values

        # Log the benchmark results
        logging.info(f"Benchmark results for {task_function.__name__}:")
        logging.info(f"Time elapsed: {time_elapsed:.2f} s")
        logging.info(f"CPU used: {cpu_used:.2f} %")
        logging.info(f"Memory used: {memory_used / (1024**2):.2f} MB")

        # Format the benchmark results
        benchmark_results = {
            "time_elapsed": f"{time_elapsed:.2f} s",
            "cpu_used": f"{cpu_used:.2f} %",
            "memory_used": f"{memory_used / (1024**2):.2f} MB"
        }

        return result, benchmark_results

    except Exception as e:
        # Log the exception with traceback
        logging.exception(f"Error during benchmarking: {e}")
        raise


    except Exception as e:
        # Log the exception with traceback
        logging.exception(f"Error during benchmarking: {e}")
        raise


@app.post("/download_and_transcribe/")
async def download_and_transcribe(youtube_urls: List[str]):
    results = {}
    for url in youtube_urls:
        try:
            # Call the async benchmark function and await its result for each URL
            response, benchmark_results = await async_benchmark(_download_and_transcribe, url)
            results[url] = {
                "file_response": response,
                "benchmark_results": benchmark_results
            }
        except Exception as e:
            logging.exception(f"Failed to process {url}: {e}")
            results[url] = {
                "error": "An error occurred.",
                "error_details": str(e)
            }
    return results
    
async def _download_and_transcribe(youtube_url: str):
    """
    Downloads audio from a YouTube URL, transcribes it, and saves the transcript.
    """
    try:
        
        yt = YouTube(youtube_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        audio_file_name = audio_stream.download(downloads_path)
        base, _ = os.path.splitext(audio_file_name)
        wav_file_path = base + '.wav'
        conversion_command = ['ffmpeg', '-y', '-i', audio_file_name, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', wav_file_path]
        conversion_process = subprocess.run(conversion_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if conversion_process.returncode != 0:
            error_details = conversion_process.stderr.decode('utf-8')
            raise HTTPException(status_code=500, detail=f"Error converting file to WAV: {error_details}")

        with wave.open(wav_file_path, 'rb') as audio_wave:
            audio_data = np.frombuffer(audio_wave.readframes(audio_wave.getnframes()), np.int16)
            text = model.stt(audio_data)

        transcript_file_name = os.path.basename(base).replace(' ', '_') + '.txt'
        transcript_file_path = os.path.join(content_directory, transcript_file_name)
        with open(transcript_file_path, 'w') as file:
            file.write(text)
        return FileResponse(path=transcript_file_path, filename=transcript_file_name, media_type='text/plain')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
def calculate_average_benchmark(benchmark_results_list):
    # Initialize totals to 0
    total_time = 0
    total_cpu = 0
    total_memory = 0
    count = len(benchmark_results_list)

    # Sum the benchmark results after converting them to appropriate types
    for result in benchmark_results_list:
        # Strip the units (e.g., " s", " %", " MB") and convert to float
        total_time += float(result['time_elapsed'].rstrip(' s'))
        total_cpu += float(result['cpu_used'].rstrip(' %'))
        total_memory += float(result['memory_used'].rstrip(' MB'))
    
    # Calculate averages
    average_benchmark = {
        "average_time_elapsed": f"{total_time / count:.2f} s" if count else "N/A",
        "average_cpu_used": f"{total_cpu / count:.2f} %" if count else "N/A",
        "average_memory_used": f"{total_memory / count:.2f} MB" if count else "N/A"
    }

    return average_benchmark


@app.post("/download_and_transcribe_multiple/")
async def download_and_transcribe_multiple(youtube_urls: List[str]):
    individual_results = []
    for url in youtube_urls:
        try:
            # Call the async benchmark function and await its result for each URL
            transcript_path, benchmark_results = await async_benchmark(_download_and_transcribe, url)
            individual_results.append(benchmark_results)
            # Store the results per URL or handle them as needed
        except Exception as e:
            logging.exception(f"Failed to process {url}: {e}")
            # Handle the error for each URL as needed

    # Calculate the average benchmark now that all individual results are collected
    average_benchmark = calculate_average_benchmark([res for res in individual_results if res])

    # Return or log the average benchmark along with individual results
    return {
        "individual_results": individual_results,
        "average_benchmark": average_benchmark
    }
    


def plot_benchmark_results(benchmark_results_list):
    # Assuming 'benchmark_results_list' is a list of dictionaries with CPU, Memory, and Video Length data
    # Normalize data
    normalized_data = []
    for result in benchmark_results_list:
        normalized_time = float(result['time_elapsed'].rstrip(' s'))
        normalized_cpu = float(result['cpu_used'].rstrip(' %'))
        # Prevent negative memory usage by setting a floor of 0
        normalized_memory = max(float(result['memory_used'].rstrip(' MB')), 0)
        normalized_data.append((normalized_time, normalized_cpu, normalized_memory))

    # Convert to NumPy array for easy column slicing
    normalized_array = np.array(normalized_data)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot CPU usage
    cpu_line, = plt.plot(normalized_array[:, 0], normalized_array[:, 1], label='CPU Usage (%)', color='g', marker='o')

    # Plot Memory usage
    memory_line, = plt.plot(normalized_array[:, 0], normalized_array[:, 2], label='Memory Usage (MB)', color='b', marker='x')

    plt.title('Resource Usage over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Resource Usage')
    plt.legend(handles=[cpu_line, memory_line])

    # Save the plot to a BytesIO buffer
    plot_buf = io.BytesIO()
    plt.savefig(plot_buf, format='png')
    plot_buf.seek(0)

    return plot_buf

@app.get("/plot_benchmark/")
async def api_plot_benchmark():
    # Call the download_and_transcribe_multiple endpoint to get benchmark results
      # Replace with the list of YouTube URLs you want to use for benchmarking
    response = await download_and_transcribe_multiple(youtube_urls)
    
    # Extract the benchmark results (individual_results) from the response
    individual_results = response.get("individual_results", [])

    # Call the plot_benchmark_results function with the benchmark results
    plot_buf = plot_benchmark_results(individual_results)

    # Create a StreamingResponse to return the plot as an image
    return StreamingResponse(plot_buf, media_type='image/png')
    
#     # Assuming 'benchmark_results' is a list of dictionaries with CPU, Memory, and Video Length data
# benchmark_results = [
#     {"time_elapsed": 91.75, "cpu_used": 9.80, "memory_used": 105.52, "video_length": 300},
#     {"time_elapsed": 113.72, "cpu_used": 11.30, "memory_used": 96.24, "video_length": 500},
#     {"time_elapsed": 134.32, "cpu_used": 18.80, "memory_used": 44.25, "video_length": 450},
#     {"time_elapsed": 151.25, "cpu_used": 6.90, "memory_used": -34.89, "video_length": 600},
#     {"time_elapsed": 118.37, "cpu_used": 10.90, "memory_used": -20.88, "video_length": 400}
# ]

# # Normalize data
# normalized_data = []
# for result in benchmark_results:
#     normalized_time = result['time_elapsed'] / result['video_length']
#     normalized_cpu = result['cpu_used'] / result['video_length']
#     # Prevent negative memory usage by setting a floor of 0
#     normalized_memory = max(result['memory_used'], 0) / result['video_length']
#     normalized_data.append((normalized_time, normalized_cpu, normalized_memory))

# # Convert to NumPy array for easy column slicing
# normalized_array = np.array(normalized_data)

# # Create the plot
# plt.figure(figsize=(10, 6))

# # Plot CPU usage
# cpu_line, = plt.plot(normalized_array[:, 1], label='CPU Usage (%)', color='g', marker='o')

# # Plot Memory usage
# memory_line, = plt.plot(normalized_array[:, 2], label='Memory Usage (MB)', color='b', marker='x')

# plt.title('Normalized Resource Usage per Second for Each Task')
# plt.xlabel('Task')
# plt.ylabel('Normalized Resource Usage per Second')
# plt.xticks(ticks=np.arange(len(benchmark_results)), labels=[f'Task {i+1}' for i in range(len(benchmark_results))])
# plt.legend(handles=[cpu_line, memory_line])

# plt.show()

# @app.post("/download_and_transcribe_multiple/")
# async def download_and_transcribe_multiple(youtube_urls: List[str]):
#     results = []
#     total_time_elapsed = 0
#     total_cpu_used = 0
#     total_memory_used = 0
#     processed_count = 0
    
#     for url in youtube_urls:
#         try:
#             transcript_path, benchmark_results = await async_benchmark(_download_and_transcribe, url)
#             results.append({
#                 "transcript_path": transcript_path,
#                 "benchmark_results": benchmark_results
#             })
#             total_time_elapsed += float(benchmark_results["time_elapsed"].rstrip(' s'))
#             total_cpu_used += float(benchmark_results["cpu_used"].rstrip(' %'))
#             total_memory_used += float(benchmark_results["memory_used"].rstrip(' MB'))
#             processed_count += 1
#         except Exception as e:
#             logging.exception(f"Failed to process {url}: {e}")
#             results.append({
#                 "url": url,
#                 "error": "An error occurred.",
#                 "error_details": str(e)
#             })

    # Calculate averages
    # avg_time_elapsed = total_time_elapsed / processed_count if processed_count > 0 else 0
    # avg_cpu_used = total_cpu_used / processed_count if processed_count > 0 else 0
    # avg_memory_used = total_memory_used / processed_count if processed_count > 0 else 0
    
    # # Compile the final report
    # final_report = {
    #     "individual_results": results,
    #     "average_benchmark": {
    #         "average_time_elapsed": f"{avg_time_elapsed:.2f} s",
    #         "average_cpu_used": f"{avg_cpu_used:.2f} %",
    #         "average_memory_used": f"{avg_memory_used:.2f} MB"
    #     }
    # }
    
    # return final_report


# def calculate_average_benchmark(benchmark_results_list):
#     total_time = sum(result['time_elapsed'] for result in benchmark_results_list)
#     total_cpu = sum(result['cpu_used'] for result in benchmark_results_list)
#     total_memory = sum(result['memory_used'] for result in benchmark_results_list)
#     count = len(benchmark_results_list)

#     average_benchmark = {
#         "average_time_elapsed": f"{total_time / count:.2f} s",
#         "average_cpu_used": f"{total_cpu / count:.2f} %",
#         "average_memory_used": f"{total_memory / count / (1024**2):.2f} MB"
#     }

#     return average_benchmark




