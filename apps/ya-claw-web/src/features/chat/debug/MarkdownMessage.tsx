import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

import { cn } from '../../../lib/utils'
import { remarkNormalizeRouteHeadings } from '../markdownHeadings'

export function MarkdownMessage({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm, remarkNormalizeRouteHeadings]}
      components={{
        a: ({ className, ...props }) => (
          <a
            className={cn(
              'font-medium text-blue-600 underline decoration-blue-300 underline-offset-2 hover:text-blue-700',
              className,
            )}
            target="_blank"
            rel="noreferrer"
            {...props}
          />
        ),
        blockquote: ({ className, ...props }) => (
          <blockquote
            className={cn(
              'my-4 border-l-4 border-slate-200 pl-4 text-slate-600',
              className,
            )}
            {...props}
          />
        ),
        code: ({ className, children, ...props }) => (
          <code
            className={cn(
              'rounded bg-slate-100 px-1.5 py-0.5 font-mono text-[0.9em] text-slate-800',
              className,
            )}
            {...props}
          >
            {children}
          </code>
        ),
        h1: ({ className, ...props }) => (
          <h2
            className={cn(
              'mb-3 mt-5 text-xl font-semibold text-slate-950',
              className,
            )}
            {...props}
          />
        ),
        h2: ({ className, ...props }) => (
          <h2
            className={cn(
              'mb-3 mt-5 text-xl font-semibold text-slate-950',
              className,
            )}
            {...props}
          />
        ),
        h3: ({ className, ...props }) => (
          <h3
            className={cn(
              'mb-3 mt-5 text-lg font-semibold text-slate-950',
              className,
            )}
            {...props}
          />
        ),
        h4: ({ className, ...props }) => (
          <h4
            className={cn(
              'mb-2 mt-4 text-base font-semibold text-slate-950',
              className,
            )}
            {...props}
          />
        ),
        h5: ({ className, ...props }) => (
          <h5
            className={cn(
              'mb-2 mt-4 text-sm font-semibold text-slate-950',
              className,
            )}
            {...props}
          />
        ),
        h6: ({ className, ...props }) => (
          <h6
            className={cn(
              'mb-2 mt-4 text-xs font-semibold uppercase tracking-wide text-slate-800',
              className,
            )}
            {...props}
          />
        ),
        li: ({ className, ...props }) => (
          <li className={cn('pl-1', className)} {...props} />
        ),
        ol: ({ className, ...props }) => (
          <ol
            className={cn('my-3 list-decimal space-y-1 pl-6', className)}
            {...props}
          />
        ),
        p: ({ className, ...props }) => (
          <p
            className={cn('my-3 leading-7 first:mt-0 last:mb-0', className)}
            {...props}
          />
        ),
        pre: ({ className, ...props }) => (
          <pre
            className={cn(
              'scrollbar-thin my-4 max-w-full overflow-auto rounded-xl border border-slate-200 bg-slate-950 p-3 text-xs leading-5 text-slate-100',
              className,
            )}
            {...props}
          />
        ),
        table: ({ className, ...props }) => (
          <div className="scrollbar-thin my-4 overflow-auto">
            <table
              className={cn(
                'w-full border-collapse text-left text-sm',
                className,
              )}
              {...props}
            />
          </div>
        ),
        td: ({ className, ...props }) => (
          <td
            className={cn(
              'border border-slate-200 px-3 py-2 align-top',
              className,
            )}
            {...props}
          />
        ),
        th: ({ className, ...props }) => (
          <th
            className={cn(
              'border border-slate-200 bg-slate-50 px-3 py-2 font-semibold',
              className,
            )}
            {...props}
          />
        ),
        ul: ({ className, ...props }) => (
          <ul
            className={cn('my-3 list-disc space-y-1 pl-6', className)}
            {...props}
          />
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  )
}
