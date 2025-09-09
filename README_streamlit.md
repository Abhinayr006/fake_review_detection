# Fake Review Detection Web App

A modern web application built with Streamlit for detecting fake reviews using machine learning.

## 🚀 Features

### Single Review Analysis
- Analyze individual reviews with detailed breakdown
- Real-time prediction with confidence scores
- Feature analysis and sentiment scoring
- Interactive input for behavioral features

### Batch Analysis
- Upload CSV files for bulk review analysis
- Progress tracking during processing
- Download results as CSV
- Summary statistics and visualizations

### Model Performance Dashboard
- Detailed accuracy metrics (72.23% accuracy)
- Confusion matrix visualization
- Feature importance analysis
- Performance breakdown by category

### About Section
- Technical details and methodology
- Dataset information
- Model architecture explanation

## 🛠️ Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK Data** (if not already done)
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"
   ```

3. **Ensure Model Files Exist**
   Make sure you have trained the model first:
   ```bash
   python main.py
   ```
   This should create `hybrid_model.pkl` and `tfidf_vectorizer.pkl`

## 🎯 Usage

1. **Start the Application**
   ```bash
   streamlit run app.py
   ```

2. **Access the Web App**
   Open your browser and go to: `http://localhost:8501`

## 📊 Interface Overview

### Navigation Sidebar
- **Single Review Analysis**: Analyze one review at a time
- **Batch Analysis**: Process multiple reviews from CSV
- **Model Performance**: View detailed metrics and charts
- **About**: Learn about the system and methodology

### Single Review Analysis Page
- **Review Input**: Text area for entering review content
- **Behavioral Features**: Sliders and inputs for user characteristics
- **Analysis Results**: Prediction with confidence and feature breakdown
- **Visual Feedback**: Color-coded results (green for genuine, red for fake)

### Batch Analysis Page
- **File Upload**: Drag and drop CSV files
- **Data Preview**: View uploaded data before processing
- **Progress Tracking**: Real-time progress bar during analysis
- **Results Download**: Export analysis results as CSV

## 🔧 Technical Details

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Features**: Hybrid (Text + Behavioral)
- **Accuracy**: 72.23% on test dataset
- **Training Data**: 500K+ Yelp reviews + labeled deceptive dataset

### Key Features Analyzed
- **Textual**: TF-IDF vectors, sentiment scores, text statistics
- **Behavioral**: User review count, average ratings, elite status
- **Metadata**: Review length, punctuation patterns, timing

### Dependencies
- `streamlit`: Web framework
- `scikit-learn`: Machine learning
- `pandas`: Data manipulation
- `nltk`: Natural language processing
- `matplotlib` & `seaborn`: Data visualization
- `wordcloud`: Text visualization

## 📈 Performance Metrics

- **Accuracy**: 72.23%
- **Precision**: 72.0%
- **Recall**: 72.0%
- **F1-Score**: 72.0%
- **Cross-validation Score**: 72.43%

## 🎨 Customization

### Styling
The app uses custom CSS for enhanced visual appeal:
- Color-coded prediction results
- Responsive design
- Professional metric cards
- Interactive elements

### Extending Functionality
- Add more visualization types
- Implement real-time model updates
- Add user authentication
- Integrate with external APIs

## 🚨 Troubleshooting

### Common Issues

1. **Model files not found**
   - Solution: Run `python main.py` first to train the model

2. **NLTK data missing**
   - Solution: Run the NLTK download commands above

3. **Port already in use**
   - Solution: Use `streamlit run app.py --server.port 8502`

4. **Memory issues with large CSV files**
   - Solution: Process files in smaller batches

### Performance Tips
- Use smaller batch sizes for better responsiveness
- Close unused browser tabs to free memory
- Consider using a more powerful machine for large datasets

## 📝 File Structure

```
fake_review_detection/
├── app.py                 # Main Streamlit application
├── main.py               # Model training pipeline
├── prediction.py         # Prediction utilities
├── data_preprocessing.py # Text preprocessing
├── data_preprocessing_yelp.py # Yelp data loading
├── requirements.txt      # Python dependencies
├── hybrid_model.pkl      # Trained model
├── tfidf_vectorizer.pkl  # TF-IDF vectorizer
└── README_streamlit.md   # This documentation
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the model training logs
3. Ensure all dependencies are properly installed
4. Check that model files exist and are readable

---

**Built with ❤️ using Streamlit and scikit-learn**
