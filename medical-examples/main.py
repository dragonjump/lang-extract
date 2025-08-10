import os
import langextract as lx

def main():
    """Main function to run medical text extraction"""
    
    # Text with interleaved medication mentions
    input_text = """The patient was prescribed Lisinopril and Metformin last month. He takes the Lisinopril 10mg daily for hypertension, but often misses his Metformin 500mg dose which should be taken twice daily for diabetes."""
    
    # Define extraction prompt
    prompt_description = """Extract medications with their details, using attributes to group related information:
1. Extract entities in the order they appear in the text
2. Each entity must have a 'medication_group' attribute linking it to its medication
3. All details about a medication should share the same medication_group value"""
    
    # Define example data with medication groups
    examples = [
        lx.data.ExampleData(
            text="Patient takes Aspirin 100mg daily for heart health and Simvastatin 20mg at bedtime.",
            extractions=[
                # First medication group
                lx.data.Extraction(
                    extraction_class="medication",
                    extraction_text="Aspirin",
                    attributes={"medication_group": "Aspirin"}  # Group identifier
                ),
                lx.data.Extraction(
                    extraction_class="dosage",
                    extraction_text="100mg",
                    attributes={"medication_group": "Aspirin"}
                ),
                lx.data.Extraction(
                    extraction_class="frequency",
                    extraction_text="daily",
                    attributes={"medication_group": "Aspirin"}
                ),
                lx.data.Extraction(
                    extraction_class="condition",
                    extraction_text="heart health",
                    attributes={"medication_group": "Aspirin"}
                ),
                # Second medication group
                lx.data.Extraction(
                    extraction_class="medication",
                    extraction_text="Simvastatin",
                    attributes={"medication_group": "Simvastatin"}
                ),
                lx.data.Extraction(
                    extraction_class="dosage",
                    extraction_text="20mg",
                    attributes={"medication_group": "Simvastatin"}
                ),
                lx.data.Extraction(
                    extraction_class="frequency",
                    extraction_text="at bedtime",
                    attributes={"medication_group": "Simvastatin"}
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
        
        print("Starting medical text extraction...")
        
        # Perform extraction
        result = lx.extract(
            text_or_documents=input_text,
            prompt_description=prompt_description,
            examples=examples,
            model_id="gemini-2.5-pro",
            api_key=api_key
        )
        
        # Display grouped medications
        print(f"\nInput text: {input_text.strip()}\n")
        print("Extracted Medications:")
        
        # Group by medication
        medication_groups = {}
        for extraction in result.extractions:
            if not extraction.attributes or "medication_group" not in extraction.attributes:
                print(f"Warning: Missing medication_group for {extraction.extraction_text}")
                continue
            
            group_name = extraction.attributes["medication_group"]
            medication_groups.setdefault(group_name, []).append(extraction)
        
        # Print each medication group
        for med_name, extractions in medication_groups.items():
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
            output_name="medical_ner_extraction.jsonl",
            output_dir="."
        )
        
        # Generate visualization
        print("Generating visualization...")
        html_content = lx.visualize("medical_ner_extraction.jsonl")
        
        with open("medical_relationship_visualization.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print("✅ Extraction completed successfully!")
        print("📁 Results saved to: medical_ner_extraction.jsonl")
        print("🌐 Visualization saved to: medical_relationship_visualization.html")
        
    except Exception as e:
        print(f"❌ Error during extraction: {str(e)}")
        print("Please check your API key and internet connection.")

if __name__ == "__main__":
    main() 