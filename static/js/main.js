document.addEventListener('DOMContentLoaded', () => {
    // 1. Dark/Light Mode Theme Toggle Setup
    const themeToggleBtn = document.getElementById('theme-toggle');
    const htmlElement = document.documentElement;
    
    // Check localStorage for saved theme preferences
    const savedTheme = localStorage.getItem('theme') || 'dark';
    htmlElement.setAttribute('data-theme', savedTheme);
    updateThemeButtonIcon(savedTheme);
    
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = htmlElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            htmlElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateThemeButtonIcon(newTheme);
        });
    }
    
    function updateThemeButtonIcon(theme) {
        if (!themeToggleBtn) return;
        if (theme === 'dark') {
            themeToggleBtn.innerHTML = '<i class="fas fa-sun"></i> Light Mode';
        } else {
            themeToggleBtn.innerHTML = '<i class="fas fa-moon"></i> Dark Mode';
        }
    }
    
    // 2. Prediction Form Submission with Spinner Animation
    const predictForm = document.getElementById('predict-form');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    if (predictForm && loadingOverlay) {
        predictForm.addEventListener('submit', (e) => {
            // Basic Frontend Validation
            const income = parseFloat(document.getElementById('AMT_INCOME_TOTAL').value);
            const age = parseFloat(document.getElementById('AGE_YEARS').value);
            const tenure = parseFloat(document.getElementById('EMPLOYMENT_YEARS').value);
            const children = parseInt(document.getElementById('CNT_CHILDREN').value);
            const members = parseInt(document.getElementById('CNT_FAM_MEMBERS').value);
            
            let isValid = true;
            let errorMsg = "";
            
            if (isNaN(income) || income <= 0) {
                isValid = false;
                errorMsg = "Income must be a positive number.";
            } else if (isNaN(age) || age < 18 || age > 100) {
                isValid = false;
                errorMsg = "Age must be between 18 and 100.";
            } else if (isNaN(tenure) || tenure < 0 || tenure > age - 15) {
                isValid = false;
                errorMsg = "Employment years must be positive and less than Age - 15.";
            } else if (isNaN(children) || children < 0) {
                isValid = false;
                errorMsg = "Children count cannot be negative.";
            } else if (isNaN(members) || members < 1 || members < children + 1) {
                isValid = false;
                errorMsg = "Family members must be at least children + 1.";
            }
            
            if (!isValid) {
                e.preventDefault();
                alert(errorMsg);
                return;
            }
            
            // Show loading overlay
            loadingOverlay.style.display = 'flex';
        });
    }
    
    // 3. Dynamic Result Gauge Rendering
    const gaugeFill = document.querySelector('.gauge-fill');
    if (gaugeFill) {
        const confidenceVal = parseFloat(gaugeFill.getAttribute('data-confidence'));
        // Circumference of circle with r=60 is 2 * PI * r = ~376.99
        const maxOffset = 376.99;
        const fillPercent = confidenceVal / 100;
        const offset = maxOffset - (fillPercent * maxOffset);
        
        // Timeout to animate the gauge rendering
        setTimeout(() => {
            gaugeFill.style.strokeDashoffset = offset;
        }, 150);
    }
    
    // 4. Form Auto-adjustments based on Unemployed select
    const isUnemployedSelect = document.getElementById('IS_UNEMPLOYED');
    const employmentYearsInput = document.getElementById('EMPLOYMENT_YEARS');
    const occupationSelect = document.getElementById('OCCUPATION_TYPE');
    
    if (isUnemployedSelect && employmentYearsInput && occupationSelect) {
        isUnemployedSelect.addEventListener('change', () => {
            if (isUnemployedSelect.value === "1") {
                employmentYearsInput.value = "0";
                employmentYearsInput.setAttribute('disabled', 'true');
                occupationSelect.value = "Pensioner";
                occupationSelect.setAttribute('disabled', 'true');
                
                // Add hidden fields so they get submitted anyway
                createOrUpdateHiddenField('EMPLOYMENT_YEARS', '0');
                createOrUpdateHiddenField('OCCUPATION_TYPE', 'Pensioner');
            } else {
                employmentYearsInput.removeAttribute('disabled');
                occupationSelect.removeAttribute('disabled');
                removeHiddenField('EMPLOYMENT_YEARS');
                removeHiddenField('OCCUPATION_TYPE');
                
                if (occupationSelect.value === "Pensioner") {
                    occupationSelect.value = "Other/Unspecified";
                }
            }
        });
        
        // Run once on load
        if (isUnemployedSelect.value === "1") {
            employmentYearsInput.value = "0";
            employmentYearsInput.setAttribute('disabled', 'true');
            occupationSelect.value = "Pensioner";
            occupationSelect.setAttribute('disabled', 'true');
            createOrUpdateHiddenField('EMPLOYMENT_YEARS', '0');
            createOrUpdateHiddenField('OCCUPATION_TYPE', 'Pensioner');
        }
    }
    
    function createOrUpdateHiddenField(name, value) {
        let hiddenInput = document.querySelector(`input[type="hidden"][name="${name}"]`);
        if (!hiddenInput) {
            hiddenInput = document.createElement('input');
            hiddenInput.type = 'hidden';
            hiddenInput.name = name;
            predictForm.appendChild(hiddenInput);
        }
        hiddenInput.value = value;
    }
    
    function removeHiddenField(name) {
        const hiddenInput = document.querySelector(`input[type="hidden"][name="${name}"]`);
        if (hiddenInput) {
            hiddenInput.parentNode.removeChild(hiddenInput);
        }
    }
});
