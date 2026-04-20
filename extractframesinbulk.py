import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def __main__():

    VIDEO_DIR = 'videos/under/'
    FRAMES_DIR = 'frames/under/'

    # list of all video file names in the VIDEO_DIR
    all_filenames_list: list[str] = os.listdir(VIDEO_DIR)

    # list to contain the video file names that havent been extracted yet
    to_extract_list: list[str] = all_filenames_list.copy()

    for filename in all_filenames_list:

        # for now keep videos from the most recent data collection session only
        should_skip = 'P035' not in filename

        # check which videos have already been extracted and remove them from the to_extract_list
        if not should_skip:
            should_skip = os.path.exists(os.path.join(FRAMES_DIR, filename.split('.')[0]))

        if should_skip:
            to_extract_list.remove(filename)

    tasks = [
        (filename, os.path.join(VIDEO_DIR, filename), os.path.join(FRAMES_DIR, filename.split('.')[0]))
        for filename in to_extract_list
    ]

    if not tasks:
        print('No videos to extract.')
        return

    # Use a small thread pool to coordinate ffmpeg subprocesses.
    max_workers = min(4, len(tasks))
    results: list[tuple[str, int]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_filename = {
            executor.submit(extract_frames, video_path, output_dir): filename
            for filename, video_path, output_dir in tasks
        }

        for future in as_completed(future_to_filename):
            filename = future_to_filename[future]
            try:
                status = future.result()
            except Exception:
                # Use a non-zero synthetic status for unexpected Python-side failure.
                status = 256
            results.append((filename, status))

    print('\nExtraction finished. Exit status per video:')
    failed = 0
    for filename, status in sorted(results, key=lambda item: item[0]):
        exit_code = decode_wait_status(status)
        state = 'OK' if exit_code == 0 else 'FAIL'
        if exit_code != 0:
            failed += 1
        print(f'  {filename}: {state} (raw_status={status}, exit_code={exit_code})')

    print(f'Summary: {len(results) - failed}/{len(results)} succeeded, {failed} failed.')
        


def extract_frames(video_path: str, output_dir: str) -> int:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return os.system(f'ffmpeg -i {video_path} -q:v 2 -start_number 0 {output_dir}/frame_%05d.jpg')


def decode_wait_status(status: int) -> int:
    # os.system() returns a wait status on Unix, which needs decoding.
    try:
        return os.waitstatus_to_exitcode(status)
    except ValueError:
        return status


if __name__ == "__main__":
    __main__()
