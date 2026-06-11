import React, { useEffect, useRef } from "react";
import { Dir, UP, DOWN, LEFT, RIGHT } from "@/engine";

export function useSwipe(onSwipe: (dir: Dir) => void) {
  const touchStartRef = useRef<{ x: number; y: number } | null>(null);

  const handleTouchStart = (e: React.TouchEvent) => {
    touchStartRef.current = {
      x: e.touches[0].clientX,
      y: e.touches[0].clientY,
    };
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    if (!touchStartRef.current) return;
    
    const touchEndX = e.changedTouches[0].clientX;
    const touchEndY = e.changedTouches[0].clientY;
    
    const dx = touchEndX - touchStartRef.current.x;
    const dy = touchEndY - touchStartRef.current.y;
    
    if (Math.abs(dx) > Math.abs(dy)) {
      if (Math.abs(dx) > 30) {
        if (dx > 0) onSwipe(RIGHT);
        else onSwipe(LEFT);
      }
    } else {
      if (Math.abs(dy) > 30) {
        if (dy > 0) onSwipe(DOWN);
        else onSwipe(UP);
      }
    }
    
    touchStartRef.current = null;
  };

  return { onTouchStart: handleTouchStart, onTouchEnd: handleTouchEnd };
}
