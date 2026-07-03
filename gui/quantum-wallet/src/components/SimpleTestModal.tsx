import { createPortal } from 'react-dom';

interface SimpleTestModalProps {
  tokenName: string;
  onClose: () => void;
}

export default function SimpleTestModal({ tokenName, onClose }: SimpleTestModalProps) {
  console.log('SimpleTestModal rendering for:', tokenName);

  const content = (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(255, 0, 0, 0.9)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 99999,
      }}
      onClick={() => {
        console.log('Modal background clicked');
        onClose();
      }}
    >
      <div
        style={{
          backgroundColor: 'white',
          padding: '50px',
          borderRadius: '20px',
          border: '5px solid yellow',
        }}
        onClick={(e) => {
          console.log('Modal content clicked');
          e.stopPropagation();
        }}
      >
        <h1 style={{ fontSize: '48px', color: 'black', marginBottom: '20px' }}>
          SIMPLE TEST MODAL WORKS!
        </h1>
        <h2 style={{ fontSize: '36px', color: 'blue' }}>
          Token: {tokenName}
        </h2>
        <button
          onClick={onClose}
          style={{
            marginTop: '30px',
            padding: '20px 40px',
            fontSize: '24px',
            backgroundColor: 'red',
            color: 'white',
            border: 'none',
            borderRadius: '10px',
            cursor: 'pointer',
          }}
        >
          CLOSE
        </button>
      </div>
    </div>
  );

  return createPortal(content, document.body);
}
