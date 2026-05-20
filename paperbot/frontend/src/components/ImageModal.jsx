import { X } from 'lucide-react';

export default function ImageModal({ image, onClose }) {
  if (!image) return null;

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div className="image-modal" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
        <button type="button" className="icon-btn modal-close" onClick={onClose} aria-label="Close image">
          <X size={18} />
        </button>
        <img src={image} alt="Retrieved paper figure" />
      </div>
    </div>
  );
}
