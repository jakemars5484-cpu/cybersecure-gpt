import os

DATA_FILE = "dataset.json"

def reset_data():
    if not os.path.exists(DATA_FILE):
        print("No dataset found to reset.")
        return

    confirm = input("Are you sure you want to reset? (y/n): ").strip().lower()

    if confirm == "y":
        try:
            os.remove(DATA_FILE)
            print("Dataset reset successfully.")
        except Exception as e:
            print("Error while resetting:", e)
    else:
        print("Reset cancelled.")

if __name__ == "__main__":
    reset_data()