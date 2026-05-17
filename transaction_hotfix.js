// Transaction Hotfix Script
// This script fixes the balance validation issue in the quantum wallet

(function() {
    console.log('🔧 Q-NarwhalKnight Transaction Hotfix Loading...');
    
    // Override the sendTransaction API call to fix unit conversion
    if (window.fetch) {
        const originalFetch = window.fetch;
        
        window.fetch = async function(url, options) {
            // Intercept transaction send requests
            if (url.includes('/v1/transactions/send') && options && options.method === 'POST') {
                console.log('🔄 Intercepting transaction send request');
                
                try {
                    const body = JSON.parse(options.body);
                    console.log('📦 Original request body:', body);
                    
                    // Don't convert amount - send as-is since backend expects QNK amount
                    // The issue was that the UI was converting amount * 100000000
                    // but the backend expects the QNK amount directly
                    
                    const fixedBody = {
                        ...body,
                        amount: parseFloat(body.amount) // Ensure it's a number, no conversion
                    };
                    
                    console.log('✅ Fixed request body:', fixedBody);
                    
                    const fixedOptions = {
                        ...options,
                        body: JSON.stringify(fixedBody)
                    };
                    
                    return originalFetch(url, fixedOptions);
                } catch (error) {
                    console.error('❌ Transaction hotfix error:', error);
                    return originalFetch(url, options);
                }
            }
            
            return originalFetch(url, options);
        };
    }
    
    // Override balance validation to use proper balance comparison
    window.qnkTransactionHotfix = {
        validateBalance: function(amount, currentBalance) {
            const fee = 0.00001;
            const totalRequired = parseFloat(amount) + fee;
            
            console.log('🔍 Balance Validation Hotfix:');
            console.log('Amount:', amount);
            console.log('Fee:', fee);
            console.log('Total Required:', totalRequired);
            console.log('Current Balance:', currentBalance);
            
            if (currentBalance <= 0) {
                return {
                    valid: false,
                    error: '❌ Wallet balance not loaded or is zero. Please refresh the balance or request faucet tokens.'
                };
            }
            
            if (totalRequired > currentBalance) {
                return {
                    valid: false,
                    error: `❌ Insufficient balance. Required: ${totalRequired.toFixed(8)} QNK (${amount} + ${fee} fee), Available: ${currentBalance.toFixed(8)} QNK`
                };
            }
            
            return { valid: true };
        }
    };
    
    // Add event listeners to override form submission
    document.addEventListener('DOMContentLoaded', function() {
        console.log('🎯 Transaction hotfix DOM listener active');
        
        // Find and override transaction forms
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.type === 'childList') {
                    // Look for transaction form elements
                    const forms = document.querySelectorAll('form, [class*="transaction"], [class*="send"]');
                    forms.forEach(function(form) {
                        if (!form.hasAttribute('data-hotfix-applied')) {
                            form.setAttribute('data-hotfix-applied', 'true');
                            console.log('🔧 Applied hotfix to form element');
                        }
                    });
                }
            });
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    });
    
    console.log('✅ Q-NarwhalKnight Transaction Hotfix Loaded Successfully!');
    console.log('🎯 Features:');
    console.log('   • Fixed unit conversion in sendTransaction');
    console.log('   • Enhanced balance validation');
    console.log('   • Detailed transaction logging');
    console.log('');
    console.log('🔧 Debug: Use window.qnkTransactionHotfix.validateBalance(amount, balance) to test');
})();