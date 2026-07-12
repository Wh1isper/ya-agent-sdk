import * as TabsPrimitive from '@radix-ui/react-tabs'
import {
  forwardRef,
  type ComponentPropsWithoutRef,
  type ComponentRef,
} from 'react'

import { cn } from '../../lib/utils'

export const Tabs = TabsPrimitive.Root

export const TabsList = forwardRef<
  ComponentRef<typeof TabsPrimitive.List>,
  ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cn(
      'inline-flex min-h-10 items-center gap-1 rounded-lg bg-[var(--subtle)] p-1 text-[var(--muted-foreground)]',
      className,
    )}
    {...props}
  />
))
TabsList.displayName = 'TabsList'

export const TabsTrigger = forwardRef<
  ComponentRef<typeof TabsPrimitive.Trigger>,
  ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      'inline-flex h-8 items-center justify-center whitespace-nowrap rounded-md px-3 text-sm font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)] disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-[var(--surface)] data-[state=active]:text-[var(--foreground)] data-[state=active]:shadow-sm',
      className,
    )}
    {...props}
  />
))
TabsTrigger.displayName = 'TabsTrigger'

export const TabsContent = forwardRef<
  ComponentRef<typeof TabsPrimitive.Content>,
  ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Content
    ref={ref}
    className={cn(
      'mt-4 min-w-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--focus)]',
      className,
    )}
    {...props}
  />
))
TabsContent.displayName = 'TabsContent'
