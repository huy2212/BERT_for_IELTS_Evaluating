import tensorflow as tf
import os
from transformers import BertTokenizer
import numpy as np
import gradio as gr

# Constants
MODEL_PATH = "training_bert_text/"
TOKENIZER_PATH = "bert-base-uncased"
MAX_SEQ_LEN = 512

# Load model
try:
    # Load the saved model
    model = tf.saved_model.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

# Load tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    print("Tokenizer loaded successfully")
except Exception as e:
    raise RuntimeError(f"Failed to load tokenizer from {TOKENIZER_PATH}: {e}")

def predict_score(task_type, question, essay):
    try:
        # Prepare the input text
        full_input = f"{task_type}: {question} [SEP] {essay}"
        
        # Tokenize the input - THIS IS THE KEY PART
        tokens = tokenizer(
            full_input,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding='max_length',
            return_tensors='tf'
        )
        
        # Extract just the input_ids tensor - the model expects this directly
        input_ids = tokens['input_ids']
        
        # Make prediction with the model
        # The model expects a tensor of input_ids directly, not a dictionary
        preds = model(input_ids)
        
        # Process prediction result
        if isinstance(preds, dict) and 'output_0' in preds:
            result = preds['output_0'].numpy()[0][0]
        else:
            result = preds.numpy()[0][0] if hasattr(preds, 'numpy') else preds[0][0]
        
        band_score = float(result)
        
        # Create detailed feedback based on band score
        feedback = ""
        if band_score >= 8.0:
            feedback = "Excellent! Your writing demonstrates a very high level of English proficiency."
        elif band_score >= 7.0:
            feedback = "Very good. Your writing shows good command of English with only occasional inaccuracies."
        elif band_score >= 6.0:
            feedback = "Good. Your writing is generally effective with some inaccuracies that don't impede communication."
        elif band_score >= 5.0:
            feedback = "Satisfactory. Your writing conveys the message but with notable errors."
        else:
            feedback = "Needs improvement. Consider working on grammar, vocabulary, and coherence."
        
        return f"### Predicted IELTS Band: {band_score:.1f}\n\n{feedback}"
            
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Example inputs
example_questions = [
    ["Task 1", "The graph below shows the percentage of people who had access to the Internet between 2017 and 2021 in four countries. Summarize the information by selecting and reporting the main features, and make comparisons where relevant.", 
     "The line graph illustrates the percentage of the population with Internet access in four different countries (Country A, B, C, and D) over a five-year period from 2017 to 2021.\n\nOverall, all four countries experienced an upward trend in Internet accessibility during this time, although at different rates and starting from different baseline levels. Country A consistently maintained the highest percentage of Internet users, while Country D showed the most significant improvement over the period.\n\nIn 2017, Country A already had approximately 85% of its population connected to the Internet, followed by Country B at around 65%. Countries C and D lagged considerably behind with only 45% and 30% respectively. By 2021, Country A had reached near-universal coverage at 95%, while Country B increased steadily to about 80%.\n\nThe most dramatic change occurred in Country D, which more than doubled its Internet penetration from 30% to roughly 70% over the five years, narrowing the gap with the more developed nations. Country C also showed consistent growth, rising from 45% to approximately 65% by the end of the period.\n\nInterestingly, the rate of growth in Countries A and B appeared to slow down in the later years, possibly indicating that they were approaching saturation points in their markets. In contrast, Countries C and D maintained steeper growth trajectories throughout the entire period, suggesting continued potential for expansion in Internet access."],
    
    ["Task 2", "Some people believe that teaching children at home is best for a child's development while others think that it is important for children to go to school. Discuss the advantages of both methods and give your own opinion.", 
     "In recent years, homeschooling has emerged as an alternative to traditional education, sparking debate about which approach better serves children's development. This essay examines the merits of both home education and conventional schooling before presenting my perspective.\n\nHomeschooling offers several compelling advantages. Firstly, it provides a customized learning experience tailored to each child's unique needs, interests, and learning pace. Unlike standardized school curricula, parents can adjust teaching methods and content to maximize their child's engagement and comprehension. Secondly, home education creates a safe learning environment free from negative peer pressure, bullying, or distractions that might impede learning. Furthermore, the flexible schedule allows for deeper exploration of subjects and incorporation of practical life skills often overlooked in traditional schools.\n\nConversely, conventional schooling provides benefits that are difficult to replicate at home. Schools offer social interaction with diverse peers, teaching children crucial interpersonal skills and exposing them to different perspectives and backgrounds. Additionally, qualified teachers bring specialized knowledge and pedagogical expertise across various subjects, particularly beneficial for advanced topics. Schools also provide structured environments with established routines that prepare children for future work settings and instill discipline.\n\nWhile both approaches have merit, I believe a balanced perspective is most prudent. Traditional schooling offers irreplaceable social development opportunities and exposure to diverse teaching expertise. However, parents should remain actively involved in their children's education, supplementing school learning when necessary and creating enriching learning experiences at home.\n\nIn conclusion, although homeschooling can be effective for some families under specific circumstances, traditional schooling, complemented by engaged parenting, generally provides the most comprehensive development for most children. The ideal approach may incorporate elements of both systems, with the specific balance determined by each child's individual needs and family circumstances."]
]

# Create the Gradio interface with improved styling
iface = gr.Interface(
    fn=predict_score,
    inputs=[
        gr.Dropdown(
            ["Task 1", "Task 2"], 
            label="IELTS Task Type",
            info="Task 1: Report writing based on visual information. Task 2: Essay writing on a given topic."
        ),
        gr.Textbox(
            lines=3, 
            label="Question Prompt",
            placeholder="Enter the IELTS question here..."
        ),
        gr.Textbox(
            lines=15, 
            label="Your Essay",
            placeholder="Write or paste your essay here (250-300 words for Task 2, 150-200 words for Task 1)..."
        ),
    ],
    outputs=gr.Markdown(),
    title="🌟 IELTS Writing Assessment Tool",
    description="""
    ### Evaluate your IELTS writing with AI
    
    This tool uses BERT (Bidirectional Encoder Representations from Transformers) to assess your IELTS writing task and provide an estimated band score.
    
    **IELTS Band Score Reference:**
    - **9**: Expert user - Complete mastery of English
    - **8**: Very good user - Fully operational command with only occasional inaccuracies
    - **7**: Good user - Operational command with occasional inaccuracies
    - **6**: Competent user - Generally effective command with some inaccuracies
    - **5**: Modest user - Partial command with notable errors
    - **4**: Limited user - Basic competence limited to familiar situations
    """,
    article="""
    ### Tips for IELTS Writing Success:
    
    1. **Read the question carefully** and ensure you understand what is being asked
    2. **Plan your response** before you start writing
    3. **Use a variety of vocabulary** and sentence structures
    4. **Organize your ideas** into clear paragraphs with logical flow
    5. **Manage your time effectively** - spend about 20 minutes on Task 1 and 40 minutes on Task 2
    
    *Note: This AI tool provides an estimated score and should be used for practice purposes only.*
    """,
    examples=example_questions,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    iface.launch()
