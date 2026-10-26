import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

import en from './locales/en.json';
import ar from './locales/ar.json';

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: en },
      ar: { translation: ar },
    },
    // Arabic-first per project decision: default language is ar, English
    // is an explicit toggle, not the fallback UI language.
    fallbackLng: 'ar',
    lng: 'ar',
    interpolation: { escapeValue: false },
  });

export default i18n;
