import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

interface StripeCheckoutProps {
  amount: string;
  walletAddress: string;
  onSuccess: () => void;
  onCancel: () => void;
}

declare global {
  interface Window {
    Stripe: any;
  }
}

/** Dynamically load the Stripe.js SDK on first use */
let stripeLoadPromise: Promise<void> | null = null;
function loadStripeScript(): Promise<void> {
  if (window.Stripe) return Promise.resolve();
  if (stripeLoadPromise) return stripeLoadPromise;
  stripeLoadPromise = new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'https://js.stripe.com/v3/';
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error('Failed to load Stripe.js'));
    document.head.appendChild(script);
  });
  return stripeLoadPromise;
}

const StripeCheckout: React.FC<StripeCheckoutProps> = ({ amount, walletAddress, onSuccess, onCancel }) => {
  const [stripe, setStripe] = useState<any>(null);
  const [cardElement, setCardElement] = useState<any>(null);
  const [clientSecret, setClientSecret] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [succeeded, setSucceeded] = useState(false);

  // Load Stripe SDK dynamically and initialize
  useEffect(() => {
    loadStripeScript()
      .then(() => {
        const publishableKey = import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY;
        if (!publishableKey) {
          setError('Stripe publishable key not configured');
          return;
        }
        const stripeInstance = window.Stripe(publishableKey);
        setStripe(stripeInstance);

        const elementsInstance = stripeInstance.elements();

        const cardElementInstance = elementsInstance.create('card', {
          style: {
            base: {
              fontSize: '16px',
              color: '#fff',
              '::placeholder': {
                color: 'rgba(255, 255, 255, 0.5)',
              },
              backgroundColor: 'transparent',
            },
            invalid: {
              color: '#f87171',
            },
          },
        });

        setCardElement(cardElementInstance);
      })
      .catch(() => {
        setError('Failed to load payment system. Please try again.');
      });
  }, []);

  // Mount card element
  useEffect(() => {
    if (cardElement) {
      cardElement.mount('#card-element');

      cardElement.on('change', (event: any) => {
        if (event.error) {
          setError(event.error.message);
        } else {
          setError(null);
        }
      });
    }

    return () => {
      if (cardElement) {
        cardElement.unmount();
      }
    };
  }, [cardElement]);

  // Create payment intent when component mounts
  useEffect(() => {
    const createPaymentIntent = async () => {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/payment/create-intent`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            wallet_address: walletAddress,
            amount: amount,
          }),
        });

        const data = await response.json();

        if (data.success && data.data) {
          setClientSecret(data.data.client_secret);
        } else {
          setError(data.error || 'Failed to create payment intent');
        }
      } catch (err) {
        setError('Failed to initialize payment. Please try again.');
      }
    };

    if (amount && walletAddress) {
      createPaymentIntent();
    }
  }, [amount, walletAddress]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!stripe || !cardElement || !clientSecret) {
      return;
    }

    setProcessing(true);
    setError(null);

    try {
      // Confirm card payment
      const { error: confirmError, paymentIntent } = await stripe.confirmCardPayment(clientSecret, {
        payment_method: {
          card: cardElement,
        },
      });

      if (confirmError) {
        setError(confirmError.message);
        setProcessing(false);
        return;
      }

      if (paymentIntent.status === 'succeeded') {
        // Call backend to confirm payment and credit wallet
        const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/v1/payment/confirm`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            payment_intent_id: paymentIntent.id,
            wallet_address: walletAddress,
          }),
        });

        const data = await response.json();

        if (data.success) {
          setSucceeded(true);
          setTimeout(() => {
            onSuccess();
          }, 1500);
        } else {
          setError(data.error || 'Failed to confirm payment');
        }
      }
    } catch (err) {
      setError('Payment failed. Please try again.');
    } finally {
      setProcessing(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="bg-gradient-to-br from-gray-900/95 to-gray-800/95 backdrop-blur-xl rounded-2xl p-6 border border-blue-500/20"
    >
      <div className="mb-6">
        <h3 className="text-xl font-bold text-white mb-2">Complete Payment</h3>
        <p className="text-gray-400 text-sm">Add ${amount} USD to your wallet</p>
      </div>

      {!succeeded ? (
        <form onSubmit={handleSubmit}>
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Card Details
            </label>
            <div
              id="card-element"
              className="bg-gray-800/50 border border-gray-700 rounded-lg p-4 focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500/20 transition-all"
            />
          </div>

          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg"
            >
              <p className="text-red-400 text-sm">{error}</p>
            </motion.div>
          )}

          <div className="flex gap-3">
            <button
              type="button"
              onClick={onCancel}
              disabled={processing}
              className="flex-1 px-4 py-3 rounded-lg bg-gray-700 hover:bg-gray-600 text-white font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!stripe || processing || !clientSecret}
              className="flex-1 px-4 py-3 rounded-lg bg-gradient-to-r from-blue-600 to-blue-500 hover:from-blue-500 hover:to-blue-400 text-white font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {processing ? (
                <>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                    className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                  />
                  Processing...
                </>
              ) : (
                `Pay $${amount}`
              )}
            </button>
          </div>
        </form>
      ) : (
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="text-center py-8"
        >
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 200, damping: 10 }}
            className="w-16 h-16 mx-auto mb-4 rounded-full bg-green-500/20 flex items-center justify-center"
          >
            <svg className="w-8 h-8 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </motion.div>
          <h4 className="text-xl font-bold text-white mb-2">Payment Successful!</h4>
          <p className="text-gray-400">Your wallet has been credited with ${amount} USD</p>
        </motion.div>
      )}
    </motion.div>
  );
};

export default StripeCheckout;
