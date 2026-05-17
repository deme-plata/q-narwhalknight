/**
 * Internationalization setup using i18next + react-i18next.
 *
 * Peer review: "The app is likely used in multiple languages.
 * The UI should be internationalized from the start."
 */

import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import en from './en.json';

const resources = {
  en: { translation: en },
};

i18n.use(initReactI18next).init({
  resources,
  lng: 'en',
  fallbackLng: 'en',
  interpolation: {
    escapeValue: false, // React already escapes
  },
  compatibilityJSON: 'v4', // Required for React Native
});

export default i18n;
