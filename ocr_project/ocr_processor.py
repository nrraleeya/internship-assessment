import os
from google import genai
from PIL import Image

class ReceiptProcessor:
    def __init__(self, api_key: str = None):
        
        key = api_key or os.environ.get("GEMINI_API_KEY")
        self.client = genai.Client(api_key=key)

    def get_transaction_amount(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            return "File not found"

        image = Image.open(image_path)

        prompt = (
            "Analyze this banking receipt image and find the primary transaction total or deposit amount. "
            "Return ONLY the plain numerical value with its decimal representation (e.g., 60.00 or 1000.00). "
            "Do not include any currency symbols, text headers, spaces, or conversational text."
        )

        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image, prompt]
        )

        return response.text.strip()


if __name__ == "__main__":
    API_KEY = "ASSESSMENT_API_KEY"
    
    processor = ReceiptProcessor(api_key=API_KEY)
    receipts = ["Technical Assessment/receipt1.png", "Technical Assessment/receipt2.png"]
    
    for path in receipts:
        try:
            amount = processor.get_transaction_amount(path)
            print(f"Deposit Amount for {path}: RM {amount}")
        except Exception as e:
            print(f"Error processing {path}: {e}")
