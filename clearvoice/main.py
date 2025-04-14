import os
import shutil
import subprocess

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from clearvoice import ClearVoice

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# A simple temporary directory for file I/O
TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page to navigate to different functionalities."""
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------
# Speech Enhancement Endpoints
# ---------------------------
@app.get("/speech-enhancement", response_class=HTMLResponse)
async def speech_enhancement_get(request: Request):
    # Provide available models to the template
    se_models = ['MossFormer2_SE_48K', 'FRCRN_SE_16K', 'MossFormerGAN_SE_16K']
    return templates.TemplateResponse("speech_enhancement.html", {"request": request, "models": se_models})


@app.post("/speech-enhancement", response_class=HTMLResponse)
async def speech_enhancement_post(
    request: Request,
    model: str = Form(...),
    file: UploadFile = File(...)
):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Save uploaded file to temp directory
    input_path = os.path.join(TEMP_DIR, file.filename)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Initialize ClearVoice for speech enhancement
    myClearVoice = ClearVoice(task='speech_enhancement', model_names=[model])
    output_wav = myClearVoice(input_path=input_path, online_write=False)

    # Save processed audio
    output_dir = os.path.join(TEMP_DIR, "speech_enhancement_output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"output_{model}.wav")
    myClearVoice.write(output_wav, output_path=output_path)

    # Render result page (you could optionally embed an HTML audio player)
    return templates.TemplateResponse("speech_enhancement_result.html", {
        "request": request,
        "audio_file": output_path,
        "model": model
    })


# ---------------------------
# Speech Separation Endpoints
# ---------------------------
@app.get("/speech-separation", response_class=HTMLResponse)
async def speech_separation_get(request: Request):
    return templates.TemplateResponse("speech_separation.html", {"request": request})


@app.post("/speech-separation", response_class=HTMLResponse)
async def speech_separation_post(
    request: Request,
    file: UploadFile = File(...)
):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    input_path = os.path.join(TEMP_DIR, file.filename)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # If the file is a video file (.avi), extract audio using ffmpeg
    if input_path.lower().endswith(".avi"):
        audio_path = input_path.rsplit(".", 1)[0] + ".wav"
        cmd = f"ffmpeg -i \"{input_path}\" -vn -acodec pcm_s16le -ar 16000 -ac 1 \"{audio_path}\""
        subprocess.call(cmd, shell=True)
        input_path = audio_path

    # Initialize ClearVoice for speech separation
    myClearVoice = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])
    output_wav = myClearVoice(input_path=input_path, online_write=False)

    output_dir = os.path.join(TEMP_DIR, "speech_separation_output")
    os.makedirs(output_dir, exist_ok=True)
    base_file_name = 'output_MossFormer2_SS_16K_'
    file_name = os.path.basename(input_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{base_file_name}{file_name}.wav")
    myClearVoice.write(output_wav, output_path=output_path)

    return templates.TemplateResponse("speech_separation_result.html", {
        "request": request,
        "output_dir": output_dir,
        "output_file": output_path
    })


# ---------------------------
# Target Speaker Extraction Endpoints
# ---------------------------
@app.get("/target-speaker-extraction", response_class=HTMLResponse)
async def target_speaker_extraction_get(request: Request):
    return templates.TemplateResponse("target_speaker_extraction.html", {"request": request})


@app.post("/target-speaker-extraction", response_class=HTMLResponse)
async def target_speaker_extraction_post(
    request: Request,
    file: UploadFile = File(...)
):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    input_path = os.path.join(TEMP_DIR, file.filename)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    output_dir = os.path.join(TEMP_DIR, "videos_tse_output")
    os.makedirs(output_dir, exist_ok=True)

    myClearVoice = ClearVoice(task='target_speaker_extraction', model_names=['AV_MossFormer2_TSE_16K'])
    # For target speaker extraction, we assume that the method writes the results into the output dir.
    myClearVoice(input_path=input_path, online_write=True, output_path=output_dir)

    return templates.TemplateResponse("target_speaker_extraction_result.html", {
        "request": request,
        "output_dir": output_dir
    })


# Optional: Endpoint to serve generated audio files (if needed)
@app.get("/audio/{file_name}", response_class=FileResponse)
async def get_audio(file_name: str):
    file_path = os.path.join(TEMP_DIR, "speech_enhancement_output", file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
