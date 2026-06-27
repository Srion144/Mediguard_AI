# predict.py — main endpoint called by the frontend
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io, base64, cv2, torch
from app.ml.model_loader import retina_model, DEVICE
from app.ml.preprocess import preprocess
from app.ml.gradcam import GradCAM, overlay

router = APIRouter()

SEVERITY = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

EXPLANATION_EN = {
    "No DR":         "No signs of diabetic retinopathy detected. Keep monitoring regularly.",
    "Mild":          "Mild signs of retina damage found. Please consult an eye doctor soon.",
    "Moderate":      "Moderate retina damage detected in the marked area. Doctor visit recommended.",
    "Severe":        "Severe retina damage found. Please see an eye specialist immediately.",
    "Proliferative": "Proliferative diabetic retinopathy detected. Urgent medical attention needed.",
}

EXPLANATION_HI = {
    "No DR":         "Diabetic retinopathy ke koi sanket nahi mile. Niyamit jaanch karte rahein.",
    "Mild":          "Retina mein halki kshati ke sanket mile hain. Jald netra chikitsak se milein.",
    "Moderate":      "Chihnahit kshetra mein madhyam retina kshati paai gayi. Doctor se milna zaroori hai.",
    "Severe":        "Gambhir retina kshati paai gayi. Kripya turant netra visheshagya se milein.",
    "Proliferative": "Gambhir awastha mein diabetic retinopathy paai gayi. Turant ilaaj zaroori hai.",
}

# initialize GradCAM after model is loaded
cam_engine = None
if retina_model is not None:
    cam_engine = GradCAM(retina_model, retina_model.layer4[-1])

@router.post("/predict")
async def predict(image: UploadFile = File(...)):
    if retina_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train retina_v1.pth on Kaggle first."
        )

    raw = await image.read()
    pil = Image.open(io.BytesIO(raw))
    x = preprocess(pil).to(DEVICE)

    cam, cls = cam_engine(x)
    out_img = overlay(cam, pil)
    _, buf = cv2.imencode(".png", out_img)
    heatmap_b64 = base64.b64encode(buf).decode()

    severity = SEVERITY[cls]
    risk_score = round(cls / 4, 2)

    return {
        "disease":        "diabetic_retinopathy",
        "severity":       severity,
        "risk_score":     risk_score,
        "heatmap_base64": heatmap_b64,
        "explanation_en": EXPLANATION_EN[severity],
        "explanation_hi": EXPLANATION_HI[severity],
    }