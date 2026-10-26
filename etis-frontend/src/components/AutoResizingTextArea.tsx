import { useRef, useLayoutEffect, useState, useEffect } from 'react';
import { Box, Typography } from '@mui/material';

interface AutoResizingTextAreaProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  minRows?: number;
  // Hard ceiling for the "content needs more room" overflow case (see
  // below). When omitted, falls back to a generous viewport-relative cap
  // (90% of window width, max 1000px). Pass an explicit value when this
  // field sits in a layout with a fixed column width it should never
  // exceed -- e.g. a side-by-side report layout, where overflowing could
  // visually overlap an adjacent column.
  maxWidthPx?: number;
}

/**
 * Behaves exactly like a normal full-width multiline field by default --
 * same look as a standard field, growing downward as content wraps. The
 * only special behavior: if a single unbroken line of text (e.g. one very
 * long URL with no spaces) would need MORE width than the field's normal
 * column to avoid an awkward mid-word wrap, the box widens beyond its
 * normal width -- up to maxWidthPx (or a generous viewport-relative
 * default when not given) -- rather than breaking the content mid-word.
 *
 * The normal column width is measured live via ResizeObserver, so "how
 * much room is available" tracks actual window/tab size rather than a
 * fixed pixel constant.
 *
 * Growth direction follows the page's reading direction automatically:
 * the surrounding flex container's justifyContent: 'flex-start' resolves
 * against the inherited CSS `direction` (set globally elsewhere in the
 * app based on the active language), so it extends to the right on an
 * LTR (English) page and to the left on an RTL (Arabic) page with no
 * manual language-checking needed here.
 */
export function AutoResizingTextArea({
  label,
  value,
  onChange,
  placeholder,
  minRows = 3,
  maxWidthPx,
}: AutoResizingTextAreaProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const mirrorRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(0);

  // Tracks the normal column width this field would otherwise fill, so we
  // only widen beyond it when content genuinely can't fit, and so that
  // "normal" width stays responsive to the window/tab being resized.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      setContainerWidth(entries[0].contentRect.width);
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    const mirror = mirrorRef.current;
    if (!textarea || !mirror || containerWidth === 0) return;

    // Measure the width the current text's longest unbroken line needs
    // (white-space: pre on the mirror means it never wraps, so this is
    // the natural width of whichever existing line is longest).
    mirror.textContent = value;
    const naturalWidth = mirror.offsetWidth;

    if (naturalWidth > containerWidth) {
      // Content needs more room than the normal column offers -- widen,
      // capped either by the explicit maxWidthPx prop or, if not given,
      // a generous viewport-relative default.
      // Never exceed 90% of the viewport (a 520px cap would overflow a
      // phone-width column), and never shrink below the column itself.
      const viewportCap = window.innerWidth * 0.9;
      const dynamicMax = Math.min(maxWidthPx ?? Math.min(viewportCap, 1000), viewportCap);
      const widened = Math.min(dynamicMax, naturalWidth);
      textarea.style.width = `${Math.max(containerWidth, widened)}px`;
    } else {
      // Normal case: fill the column exactly as a standard full-width
      // field would -- this is the common case for ordinary pasted text.
      textarea.style.width = '100%';
    }

    // Height: auto-size to content via scrollHeight, but never below the
    // minRows floor, so short text still shows the familiar multi-row
    // box instead of collapsing to a single line.
    const lineHeightPx = 24; // matches lineHeight: 1.5 at the 1rem font-size used below
    const verticalPaddingPx = 17; // 8.5px top + 8.5px bottom
    const minHeightPx = lineHeightPx * minRows + verticalPaddingPx;

    textarea.style.height = 'auto';
    textarea.style.height = `${Math.max(minHeightPx, textarea.scrollHeight)}px`;
  }, [value, containerWidth, minRows, maxWidthPx]);

  const sharedFontStyle = {
    fontFamily: 'inherit',
    fontSize: '1rem',
    lineHeight: 1.5,
  } as const;

  return (
    <Box>
      <Typography variant="caption" sx={{ display: 'block', mb: 0.5, color: 'text.secondary' }}>
        {label}
      </Typography>
      <Box ref={containerRef} sx={{ display: 'flex', justifyContent: 'flex-start', width: '100%' }}>
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          rows={minRows}
          style={{
            ...sharedFontStyle,
            padding: '8.5px 14px',
            border: '1px solid rgba(0, 0, 0, 0.23)',
            borderRadius: 8,
            resize: 'none',
            overflow: 'hidden',
            boxSizing: 'border-box',
            outline: 'none',
            display: 'block',
            flexShrink: 0,
            transition: 'width 0.15s ease, height 0.15s ease',
          }}
        />
      </Box>
      {/* Hidden mirror used only to measure the width the current text's
          longest unbroken line would need -- never shown, never
          interactive. */}
      <div
        ref={mirrorRef}
        aria-hidden="true"
        style={{
          ...sharedFontStyle,
          position: 'absolute',
          visibility: 'hidden',
          whiteSpace: 'pre',
          display: 'inline-block',
          padding: '8.5px 14px',
          top: 0,
          left: 0,
          pointerEvents: 'none',
        }}
      />
    </Box>
  );
}
