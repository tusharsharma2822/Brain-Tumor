import { useState } from "react";
import axios from "axios";

export default function UploadForm() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [maskURL, setMaskURL] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
    setResult(null);
    setMaskURL(null);
  };

  const handleSubmit = async () => {
    if (!image) return;
    const formData = new FormData();
    formData.append("image", image);
    setLoading(true);

    try {
      const res = await axios.post("http://localhost:5000/predict", formData);
      setResult(res.data);

      if (res.data.tumor_detected) {
        const mask = res.data.segmentation;
        const canvas = document.createElement("canvas");
        canvas.width = 128;
        canvas.height = 128;
        const ctx = canvas.getContext("2d");
        const imgData = ctx.createImageData(128, 128);

        for (let i = 0; i < 128; i++) {
          for (let j = 0; j < 128; j++) {
            const val = mask[i][j] * 255;
            const idx = (i * 128 + j) * 4;
            imgData.data[idx] = val;
            imgData.data[idx + 1] = val;
            imgData.data[idx + 2] = val;
            imgData.data[idx + 3] = 255;
          }
        }

        ctx.putImageData(imgData, 0, 0);
        setMaskURL(canvas.toDataURL());
      }
    } catch (err) {
      alert("Something went wrong!");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg text-left space-y-4">
      <input
        type="file"
        onChange={handleImageChange}
        accept="image/*"
        className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-700"
      />
      <button
        onClick={handleSubmit}
        disabled={!image || loading}
        className="w-full bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 transition"
      >
        {loading ? "Predicting..." : "Predict Tumor"}
      </button>

      {result && (
        <div className="mt-4 space-y-2">
          <p><strong>Prediction:</strong> {result.label}</p>
          <p><strong>Tumor Detected:</strong> {result.tumor_detected ? "Yes ✅" : "No ❌"}</p>
          {maskURL && (
            <div className="mt-4">
              <h3 className="text-lg mb-2">Predicted Mask:</h3>
              <img src={maskURL} alt="Tumor Mask" className="border border-gray-600" />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
