import os
import langextract as lx

def main():
    """Main function to run observation text extraction"""
    
    # Text with interleaved product mentions
    input_text = """  
- C: “Why was I charged twice this month?” A: “I see two pending authorizations; one will drop off within 24–48 hours.”
- C: “My promo code didn’t apply.” A: “It expired yesterday, but I can honor a one-time adjustment now.”
- C: “This foreign transaction fee surprised me.” A: “It’s your bank’s fee; I’ll send documentation to request a waiver.”
- C: “Tracking hasn’t updated in three days.” A: “Carrier reports weather delays; arrival now expected Friday.”
- C: “Package shows delivered, but I don’t have it.” A: “It was left at Building B mailroom; I’ll share the photo.”
- C: “Can you reroute to my office?” A: “I submitted a hold-and-pickup request at your nearest hub.”

- C: “How do I return without the original box?” A: “Use our reusable mailer; I’ll email a QR code.”
- C: “Refund’s taking too long.” A: “It’s processed on our end; banks take 5–10 business days.”
- C: “I received the wrong size.” A: “Exchange created; keep the original until the replacement arrives.”
- C: “The app keeps freezing.” A: “Update to v4.2.1; it fixes the memory leak causing freezes.”
- C: “My device won’t charge.” A: “Try a 30-second power reset and a different USB-C cable.”
- C: “Audio is distorted on calls.” A: “Disable noise suppression once; if resolved, update your drivers.”
- C: “I’m locked out after two-factor.” A: “I’ll verify identity and reset your authenticator seeds.”
- C: “Suspicious login alert?” A: “We blocked it from a new device; please rotate your password.”
- C: “How do I add a backup email?” A: “Go to Profile > Security > Recovery and add it there.”
- C: “Which plan fits a five-person team?” A: “The Pro plan with five seats and shared workspaces.”
- C: “Is there an NGO discount?” A: “Yes, 30% off with your verification letter.”
- C: “Can I lock pricing for two years?” A: “A 24-month agreement guarantees rate stability.”
- C: “I want to cancel today.” A: “I’ll process it now and prorate the remainder.”
- C: “The product’s not for me.” A: “Understood; may I suggest a lighter plan before you go?”
- C: “Fees keep adding up.” A: “I can consolidate and cap costs with a fixed-rate bundle.”
- C: “Your support was fantastic.” A: “Thank you! I’ll pass this to the team and note your feedback.”
- C: “Docs are confusing.” A: “I’ll rewrite the page with examples and video clips.”
- C: “The UI feels cluttered.” A: “We’re testing a simplified layout; want early access?”
- - C: “Is this item in stock?” A: “Yes, 27 units; I’ve reserved two for your cart.”
- C: “Preorder ETA?” A: “Batch arrives on the 18th; shipping starts the 19th.”
- C: “Can you bundle items?” A: “Created a bundle SKU with combined discount.”
- 
- C: “Can I talk to a supervisor?” A: “I’ve looped in a manager who’ll join shortly.”
- C: “I need a statement for insurance.” A: “Sent a signed PDF with serial numbers and dates.”
- C: “Do you integrate with Zapier?” A: “Yes—triggers for new orders and refunds are supported.”


    
    """
    
    # Define extraction prompt
    prompt_description = """Extract conversation with their details, using attributes to group related information:
1. Extract entities in the order they appear in the text
2. Each entity must have a 'product_group' attribute linking it to its complaints. Product groups are [Billing-Payment, Shipping-Delivery, 
Returns-Refunds, Product-Issues-Troubleshooting, Account-Access-Security, Sales-Pricing-Plans, Cancellations-Retention, 
Feedback-CX-NPS, Orders-Inventory-Fulfillment, Miscellaneous-Escalations]
3. All details about a complaints should share the same product_group value"""
    
    # Define example data with product groups
    examples = [
        lx.data.ExampleData(
            text="Customer: 'My tracking hasn't updated in three days.' Agent: 'Carrier reports weather delays; arrival now expected Friday.' Customer: 'Can I pay via bank transfer?' Agent: 'Yes, here are ACH details; include your order ID as reference.'",
            extractions=[
                # First product group - Shipping-Delivery
                lx.data.Extraction(
                    extraction_class="shipping_issue",
                    extraction_text="tracking hasn't updated in three days",
                    attributes={"product_group": "Shipping-Delivery"}
                ),
                lx.data.Extraction(
                    extraction_class="delay_reason",
                    extraction_text="weather delays",
                    attributes={"product_group": "Shipping-Delivery"}
                ),
                lx.data.Extraction(
                    extraction_class="expected_arrival",
                    extraction_text="arrival now expected Friday",
                    attributes={"product_group": "Shipping-Delivery"}
                ),
                # Second product group - Billing-Payment
                lx.data.Extraction(
                    extraction_class="payment_method_inquiry",
                    extraction_text="Can I pay via bank transfer?",
                    attributes={"product_group": "Billing-Payment"}
                ),
                lx.data.Extraction(
                    extraction_class="payment_instructions",
                    extraction_text="ACH details; include your order ID as reference",
                    attributes={"product_group": "Billing-Payment"}
                )
            ]
        ),
        lx.data.ExampleData(
            text="Customer: 'I want to return this defective product.' Agent: 'I can help with that. What's the issue?' Customer: 'It arrived damaged and doesn't work properly.' Agent: 'I'll process a return and send you a prepaid shipping label.'",
            extractions=[
                # Product group - Returns-Refunds
                lx.data.Extraction(
                    extraction_class="return_request",
                    extraction_text="I want to return this defective product",
                    attributes={"product_group": "Returns-Refunds"}
                ),
                lx.data.Extraction(
                    extraction_class="product_condition",
                    extraction_text="It arrived damaged and doesn't work properly",
                    attributes={"product_group": "Returns-Refunds"}
                ),
                lx.data.Extraction(
                    extraction_class="return_solution",
                    extraction_text="I'll process a return and send you a prepaid shipping label",
                    attributes={"product_group": "Returns-Refunds"}
                )
            ]
        )
    ]
    
    try:
        # Get API key from environment variable or prompt user
        api_key = os.getenv("LANGEXTRACT_API_KEY")
        if not api_key:
            print("Warning: LANGEXTRACT_API_KEY environment variable not set.")
            print("Please set your API key or the extraction may fail.")
            api_key = None
        
        print("Starting observation text extraction...")
        
        # Perform extraction
        result = lx.extract(
            text_or_documents=input_text,
            prompt_description=prompt_description,
            examples=examples,
            model_id="gemini-2.5-pro",
            api_key=api_key
        )
        
        # result = lx.extract(
        # text_or_documents=input_text,
        # prompt_description=prompt_description,
        # examples=examples,
        # timeout=10000,
        # language_model_type=lx.inference.OllamaLanguageModel,
        # model_id="gemma3:1b",  # Automatically selects Ollama provider
        # model_url="http://localhost:11434",
        # temperature=0.3,
        # fence_output=False,
        # use_schema_constraints=False
        # )
   
        
    
        # Display grouped products
        print(f"\nInput text: {input_text.strip()}\n")
        print("Extracted products:")
        
        # Group by product
        product_groups = {}
        for extraction in result.extractions:
            if not extraction.attributes or "product_group" not in extraction.attributes:
                print(f"Warning: Missing product_group for {extraction.extraction_text}")
                continue
            
            group_name = extraction.attributes["product_group"]
            product_groups.setdefault(group_name, []).append(extraction)
        
        # Print each product group
        for med_name, extractions in product_groups.items():
            print(f"\n* {med_name}")
            for extraction in extractions:
                position_info = ""
                if extraction.char_interval:
                    start, end = extraction.char_interval.start_pos, extraction.char_interval.end_pos
                    position_info = f" (pos: {start}-{end})"
                print(f"  • {extraction.extraction_class.capitalize()}: {extraction.extraction_text}{position_info}")
        
        # Save results
        print("\nSaving results...")
        lx.io.save_annotated_documents(
            [result],
            output_name="observation_ner_extraction.jsonl",
            output_dir="."
        )
        
        # Generate visualization
        print("Generating visualization...")
        html_content = lx.visualize("observation_ner_extraction.jsonl")
        
        with open("observation_relationship_visualization.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print("✅ Extraction completed successfully!")
        print("📁 Results saved to: observation_ner_extraction.jsonl")
        print("🌐 Visualization saved to: observation_relationship_visualization.html")
        
    except Exception as e:
        print(f"❌ Error during extraction: {str(e)}")
        print("Please check your API key and internet connection.")

if __name__ == "__main__":
    main() 