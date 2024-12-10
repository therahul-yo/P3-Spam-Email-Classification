import streamlit as st
import pickle

# Load pre-trained model and vectorizer
model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# Main function to create Streamlit UI
def main():
    # Set the title of the web application
    st.title("📝 Email Spam Classification App")
    
    # Description of the application
    st.write("""
    This Machine Learning application classifies emails as either **Spam** or **Ham** (not spam). 
    Simply enter the text of the email below, and click **Classify** to see the result.
    """)

    # Input area for the user to enter email content
    st.subheader("📥 Enter Email Content to Classify:")
    user_input = st.text_area("Type or Paste the email content here:", height=200)

    # Provide some example text for the user
    st.markdown("""
    ### Example Email Content:
    - "Congratulations, you've won a $1000 gift card! Click here to claim your prize."
    - "Dear user, your bank account has been suspended. Please verify your information to reactivate it."
    - "Reminder: Your meeting with John is scheduled for tomorrow at 3 PM."
    """)

    # Add a horizontal line for better separation
    st.markdown("---")

    # Button to classify the email
    if st.button("Classify Email"):
        if user_input:
            # Prepare the data to be classified
            data = [user_input]
            
            # Transform the email into vector form using the vectorizer
            vec = cv.transform(data).toarray()
            
            # Predict if the email is spam or not using the model
            result = model.predict(vec)
            
            # Show the result to the user with some enhancements
            if result[0] == 0:
                st.success("✅ **This is NOT a Spam Email.**")
                st.image("https://img.icons8.com/color/96/000000/checkmark.png", width=60)
            else:
                st.error("❌ **This is a Spam Email.**")
                st.image("https://img.icons8.com/color/96/000000/delete-forever.png", width=60)
        else:
            st.warning("⚠️ **Please enter the email content to classify.**")

    # Add another horizontal line for separation
    st.markdown("---")

    # Additional Information or Footer
    st.write("Created with ❤️ by [Your Name]")
    st.write("Want to learn more? Visit the [Spam Classification Project](#).")

# Run the Streamlit application
if __name__ == '__main__':
    main()
