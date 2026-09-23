document.getElementById('copy-full-prompt').addEventListener('click', async () => {
  const status = document.getElementById('copy-prompt-status');
  try {
    const response = await fetch('assets/TRAIN-MY-SPECIALIST.md');
    if (!response.ok) throw new Error('Prompt download failed');
    await navigator.clipboard.writeText(await response.text());
    status.textContent = 'Copied all eight sections. Paste them into Claude and change the bracketed fields.';
  } catch {
    status.textContent = 'Use Download the prompt, then select and copy its text.';
  }
});
