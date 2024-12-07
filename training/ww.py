import pickle
from pathlib import Path
import face_recognition
from PIL import Image, UnidentifiedImageError


class FaceTrainer:
    def __init__(self, training_dir: str, encodings_file: str, model: str = "hog"):
        self.training_dir = Path(training_dir)
        self.encodings_file = Path(encodings_file)
        self.model = model

    def _prepare_image(self, filepath: Path) -> Image.Image:
        try:
            image = Image.open(filepath)
            image = image.resize((500, 500), Image.ANTIALIAS) 
            return image
        except UnidentifiedImageError as e:
            print(f"Error loading image {filepath}: {e}")
            return None

    def train(self) -> None:
        names = []
        encodings = []

        for filepath in self.training_dir.glob("*/*"):
            try:
                name = filepath.parent.name
                image_name = filepath.stem

                pil_image = self._prepare_image(filepath)
                if pil_image is None:
                    continue

                image = face_recognition.load_image_file(filepath)
                print(f"Processing {image_name}...")

            except (FileNotFoundError, UnidentifiedImageError) as e:
                print(f"Error processing file {filepath}: {e}")
                continue

            face_locations = face_recognition.face_locations(image, model=self.model)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for encoding in face_encodings:
                names.append(name)
                encodings.append(encoding)

        name_encodings = {"names": names, "encodings": encodings}
        with self.encodings_file.open(mode="wb") as f:
            pickle.dump(name_encodings, f)
        print(f"Encodings saved to {self.encodings_file}")


if __name__ == "__main__":
    training_directory = "training/images"  
    encodings_output_file = "training_encoding.pkl" 

    trainer = FaceTrainer(training_dir=training_directory, encodings_file=encodings_output_file)
    trainer.train()
