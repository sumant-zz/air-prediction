<pre>flowchart TD
    A([Start]) --> B[Load Air Quality Data<br>for 4 Cities]
    B --> C[Preprocess Data<br>(Fill Missing, Scale Values)]
    C --> D[Split into Train/Test Sets]
    D --> E[Train Model for Each City<br>(LSTM / Random Forest)]
    E --> F[Evaluate Performance<br>on Test Set]
    F --> G{R² > 0.9 ?}
    G -->|Yes| M[Save Model<br>model_city.pkl]
    G -->|No| N[No Alert]

    M --> O[Predict Next 4 Months<br>Iterative Forecasting]
    N --> O

    O --> P[Save to .pkl Files<br>→ model_City.pkl<br>→ predictions.pkl<br>→ scaler.pkl]
    P --> Q{All 4 Cities Done?}
    Q -->|No| E
    Q -->|Yes| R[Generate .pkl Files<br>6 Files Created]

    R --> S[Launch Streamlit App<br><code>streamlit run app.py</code>]
    S --> T[Load All .pkl Files<br><code>joblib.load()</code>]
    T --> U[Display Dashboard<br>→ 4 City Risk Cards<br>→ 4-Month Charts<br>→ Interactive Slider]
    U --> V[User Selects City + Months<br>e.g., Delhi + 12 Months]
    V --> W[Predict Future Months<br>Using Saved Model + Last Sequence]
    W --> X[Show Graph + Table + Alert<br>Red if >80]
    X --> Y([End: Live AI Dashboard Running])

    %% Styling
    style A fill:#2ecc71,stroke:#27ae60,color:white
    style Y fill:#e74c3c,stroke:#c0392b,color:white
    style M fill:#e67e22,stroke:#d35400,color:white
    style S fill:#3498db,stroke:#2980b9,color:white
    style U fill:#9b59b6,stroke:#8e44ad,color:white
</pre>
