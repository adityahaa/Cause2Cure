from causal import train_causal_artifacts

if __name__ == "__main__":
    result = train_causal_artifacts()
    print(f"Causal artifacts saved for: {', '.join(result['treatments'])}")
