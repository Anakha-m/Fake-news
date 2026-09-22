/**
 * Fake News Detection System - Client-side Utilities
 */

document.addEventListener('DOMContentLoaded', function () {
  // 1. Character and Word Counter
  const newsInput = document.getElementById('newsInputText');
  const charCounter = document.getElementById('charCounter');
  const wordCounter = document.getElementById('wordCounter');
  const clearBtn = document.getElementById('clearBtn');
  const submitBtn = document.getElementById('predictSubmitBtn');
  const predictionForm = document.getElementById('predictionForm');

  function updateCounts() {
    if (!newsInput) return;
    const text = newsInput.value || '';
    const charCount = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;

    if (charCounter) charCounter.textContent = `${charCount} characters`;
    if (wordCounter) wordCounter.textContent = `${words} words`;
  }

  if (newsInput) {
    newsInput.addEventListener('input', updateCounts);
    updateCounts();
  }

  // 2. Clear Button Action
  if (clearBtn && newsInput) {
    clearBtn.addEventListener('click', function () {
      newsInput.value = '';
      newsInput.focus();
      updateCounts();
    });
  }

  // 3. Form Submission Loading Spinner
  if (predictionForm && submitBtn) {
    predictionForm.addEventListener('submit', function (e) {
      if (newsInput && newsInput.value.trim().length < 15) {
        return; // Let standard validation handle min length
      }
      submitBtn.disabled = true;
      submitBtn.innerHTML = `
        <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
        Analyzing & Verifying Evidence...
      `;
    });
  }

  // 4. Sample News Inserters for Demo
  const samplePills = document.querySelectorAll('.sample-news-pill');
  samplePills.forEach(pill => {
    pill.addEventListener('click', function (e) {
      e.preventDefault();
      const sampleText = this.getAttribute('data-sample');
      if (newsInput && sampleText) {
        newsInput.value = sampleText;
        updateCounts();
        newsInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    });
  });
});
