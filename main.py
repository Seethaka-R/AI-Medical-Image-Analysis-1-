"""
main.py — CLI for Chest X-Ray Pneumonia classifier (no dashboard).

Usage:
  python main.py --mode train
  python main.py --mode train --epochs1 15 --epochs2 10
  python main.py --mode predict --image path/to/xray.jpg
  python main.py --mode evaluate
"""
import argparse, json
from pathlib import Path

BASE     = Path(__file__).parent
DATA_DIR = str(BASE / "data" / "raw")
MODEL    = str(BASE / "models" / "saved" / "pneumonia_resnet50.h5")
OUT_DIR  = str(BASE / "outputs")


def main():
    parser = argparse.ArgumentParser(description="PneumoAI CLI")
    parser.add_argument("--mode",    choices=["train","predict","evaluate"], required=True)
    parser.add_argument("--image",   default=None, help="X-ray image path (predict mode)")
    parser.add_argument("--data",    default=DATA_DIR)
    parser.add_argument("--model",   default=MODEL)
    parser.add_argument("--epochs1", type=int, default=10)
    parser.add_argument("--epochs2", type=int, default=8)
    args = parser.parse_args()

    if args.mode == "train":
        from src.trainer import train
        print(f"\nDataset : {args.data}")
        print(f"Model   : {args.model}\n")
        result = train(
            data_dir=args.data,
            model_path=args.model,
            out_dir=OUT_DIR,
            epochs_phase1=args.epochs1,
            epochs_phase2=args.epochs2,
        )
        print("\n== Evaluation Results ==")
        print(json.dumps(result["eval_metrics"], indent=2))

    elif args.mode == "predict":
        import tensorflow as tf
        from src.predictor import predict_image
        if not args.image:
            print("ERROR: provide --image path/to/xray.jpg"); return
        print(f"Loading model from {args.model} ...")
        model  = tf.keras.models.load_model(args.model, compile=False)
        result = predict_image(model, args.image, out_dir=f"{OUT_DIR}/predictions")
        print("\n== Prediction ==")
        print(f"  Label       : {result['label']}")
        print(f"  Confidence  : {result['confidence']}%")
        print(f"  Raw score   : {result['raw_score']}  (sigmoid output)")
        print(f"  NORMAL      : {result['probabilities']['NORMAL']}%")
        print(f"  PNEUMONIA   : {result['probabilities']['PNEUMONIA']}%")
        print(f"  Risk Level  : {result['risk_level']}")
        print(f"  Grad-CAM    : {result['gradcam_path']}")
        print(f"\nRecommendation:\n  {result['recommendation']}")

    elif args.mode == "evaluate":
        import tensorflow as tf
        from src.preprocessing import get_generators
        from src.evaluate import evaluate_model
        model = tf.keras.models.load_model(args.model, compile=False)
        _, _, test_gen = get_generators(args.data)
        from src.preprocessing import CLASS_NAMES
        metrics = evaluate_model(model, test_gen, CLASS_NAMES, OUT_DIR, binary=True)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
