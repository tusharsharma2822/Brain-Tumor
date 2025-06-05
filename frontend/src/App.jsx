import UploadForm from "./components/UploadForm";

export default function App() {
  return (
    <div className="min-h-screen p-6 bg-gray-900 text-white">
      <div className="max-w-2xl mx-auto text-center">
        <h1 className="text-3xl font-bold mb-6">🧠 Brain Tumor Detector</h1>
        <UploadForm />
      </div>
    </div>
  );
}
