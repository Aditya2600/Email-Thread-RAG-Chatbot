// The only real account identity the app ever sees is the Gmail address the
// backend hands back on the OAuth callback (?email=…). We stash it so the
// sidebar can show the real connected mailbox instead of a placeholder. There
// is no display name in that data, so we never invent one.

const KEY = "inbox-copilot.account-email";

export function getAccountEmail(): string | null {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function setAccountEmail(email: string): void {
  try {
    localStorage.setItem(KEY, email);
  } catch {
    // Private mode / storage disabled: the sidebar just shows "not connected".
  }
}

/** Two-letter avatar initials from the local part, e.g. jordan@x → JO. */
export function initialsFromEmail(email: string): string {
  const local = email.split("@")[0] ?? "";
  const letters = local.replace(/[^a-zA-Z]/g, "");
  return (letters.slice(0, 2) || email.slice(0, 2)).toUpperCase();
}
